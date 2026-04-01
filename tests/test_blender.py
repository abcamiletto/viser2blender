from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from viser2blender import load_viser_recording, normalize_recording
from viser2blender._types import RecordingPayload


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MIN_BPY_VERSION = (5, 1, 0)


def test_blender_cli_validates_basic_fixture(tmp_path: Path) -> None:
    output_path = tmp_path / "basic.blend"
    manifest_path = tmp_path / "basic_manifest.json"

    result = _run_blender_cli(
        ASSETS_DIR / "blender_basic_mesh.viser",
        output_path,
        "--validate-only",
        "--emit-manifest",
        str(manifest_path),
        "--overwrite",
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["fps"] == 30.0
    assert manifest["frame_count"] == 3
    assert any(node["kind"] == "mesh" for node in manifest["nodes"])
    assert any(node["kind"] == "root" for node in manifest["nodes"])


def test_blender_cli_runs_full_conversion_with_version_aware_result(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "converted.blend"
    result = _run_blender_cli(
        ASSETS_DIR / "blender_basic_mesh.viser",
        output_path,
        "--overwrite",
    )

    bpy_version = _bpy_version()
    if bpy_version is not None and bpy_version >= MIN_BPY_VERSION:
        assert result.returncode == 0, result.stderr
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert b"BLENDER" in output_path.read_bytes()[:128]
        return

    assert result.returncode != 0
    assert "requires Blender bpy>=5.1.0" in result.stderr


def test_blender_cli_rejects_unsupported_fixture(tmp_path: Path) -> None:
    output_path = tmp_path / "unsupported.blend"

    result = _run_blender_cli(
        ASSETS_DIR / "blender_unsupported_audio.viser",
        output_path,
        "--validate-only",
    )

    assert result.returncode != 0
    assert "Unsupported message type 'AddAudioMessage'" in result.stderr


def test_normalizer_creates_explicit_group_nodes_for_path_segments() -> None:
    manifest = normalize_recording(
        load_viser_recording(ASSETS_DIR / "blender_showcase.viser")
    )
    nodes_by_name = {node.name: node for node in manifest.nodes}

    assert nodes_by_name[""].kind == "root"
    assert nodes_by_name["/group"].kind == "group"
    assert nodes_by_name["/group"].parent_name == ""
    assert nodes_by_name["/late"].kind == "group"
    assert nodes_by_name["/late"].parent_name == ""
    assert nodes_by_name["/group/mesh"].parent_name == "/group"
    assert nodes_by_name["/late/helper"].parent_name == "/late"
    assert nodes_by_name[""].transform_keyframes[0].rotation_wxyz is not None


def test_normalizer_supports_transform_only_group_nodes() -> None:
    manifest = normalize_recording(
        RecordingPayload(
            duration_seconds=1.0,
            viser_version="test",
            messages=[
                (0.0, _mesh_message("/group/mesh", [255, 0, 0])),
                (0.0, _position_message("/group", [1.0, 2.0, 3.0])),
            ],
        )
    )
    nodes_by_name = {node.name: node for node in manifest.nodes}

    assert nodes_by_name["/group"].kind == "group"
    assert nodes_by_name["/group"].transform_keyframes[0].position == [1.0, 2.0, 3.0]
    assert nodes_by_name["/group/mesh"].parent_name == "/group"


def test_normalizer_allows_path_recreation_after_removal() -> None:
    manifest = normalize_recording(
        RecordingPayload(
            duration_seconds=1.0,
            viser_version="test",
            messages=[
                (0.0, _mesh_message("/mesh", [255, 0, 0])),
                (0.5, {"type": "RemoveSceneNodeMessage", "name": "/mesh"}),
                (1.0, _mesh_message("/mesh", [0, 255, 0])),
            ],
        )
    )
    mesh_nodes = [node for node in manifest.nodes if node.name == "/mesh"]

    assert len(mesh_nodes) == 2
    assert mesh_nodes[0].node_id == "/mesh#0"
    assert mesh_nodes[0].destroy_frame == 2
    assert mesh_nodes[1].node_id == "/mesh#1"
    assert mesh_nodes[1].create_frame == 3
    assert mesh_nodes[1].destroy_frame is None


def test_remove_cascades_to_descendants() -> None:
    manifest = normalize_recording(
        RecordingPayload(
            duration_seconds=0.5,
            viser_version="test",
            messages=[
                (0.0, _mesh_message("/group/mesh", [255, 0, 0])),
                (0.5, {"type": "RemoveSceneNodeMessage", "name": "/group"}),
            ],
        )
    )
    nodes_by_name = {node.name: node for node in manifest.nodes}

    assert nodes_by_name["/group"].destroy_frame == 2
    assert nodes_by_name["/group/mesh"].destroy_frame == 2
    assert nodes_by_name["/group/mesh"].visibility_keyframes[-1].visible is False


def test_duration_extends_frame_count_even_without_terminal_updates() -> None:
    manifest = normalize_recording(
        RecordingPayload(
            duration_seconds=1.0,
            viser_version="test",
            messages=[
                (0.0, _mesh_message("/mesh", [255, 0, 0])),
                (0.5, _position_message("/mesh", [1.0, 0.0, 0.0])),
            ],
        )
    )

    assert manifest.fps == 2.0
    assert manifest.frame_count == 3


def _mesh_message(name: str, color: list[int]) -> dict[str, object]:
    return {
        "type": "MeshMessage",
        "name": name,
        "props": {
            "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "faces": [[0, 1, 2]],
            "color": color,
            "scale": 1.0,
            "wireframe": False,
            "opacity": None,
            "flat_shading": False,
            "side": "front",
            "material": "standard",
            "cast_shadow": True,
            "receive_shadow": True,
        },
    }


def _position_message(name: str, position: list[float]) -> dict[str, object]:
    return {
        "type": "SetPositionMessage",
        "name": name,
        "position": position,
    }


def _run_blender_cli(
    input_path: Path,
    output_path: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "viser2blender",
        str(input_path),
        str(output_path),
        *extra_args,
    ]
    return subprocess.run(command, capture_output=True, text=True)


def _bpy_version() -> tuple[int, int, int] | None:
    if importlib.util.find_spec("bpy") is None:
        return None
    bpy = importlib.import_module("bpy")
    return tuple(bpy.app.version)
