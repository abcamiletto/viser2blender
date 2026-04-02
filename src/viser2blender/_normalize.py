from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, cast

import numpy as np

from ._types import (
    ColorValue,
    CylinderGeometry,
    FrameGeometry,
    GeometryData,
    GeometryKeyframe,
    IcosphereGeometry,
    LineSegmentsGeometry,
    MaterialStyle,
    MeshGeometry,
    NodeManifest,
    PointCloudGeometry,
    RecordingManifest,
    RecordingPayload,
    TransformKeyframe,
    VisibilityKeyframe,
)

ROOT_NODE_ID = "#0"


SUPPORTED_MESSAGE_TYPES = {
    "RunJavascriptMessage",
    "BackgroundImageMessage",
    "FrameMessage",
    "MeshMessage",
    "BoxMessage",
    "PointCloudMessage",
    "LineSegmentsMessage",
    "IcosphereMessage",
    "CylinderMessage",
    "SetGuiPanelLabelMessage",
    "SetPositionMessage",
    "SetOrientationMessage",
    "SetSceneNodeVisibilityMessage",
    "RemoveSceneNodeMessage",
    "SceneNodeUpdateMessage",
}

CREATE_MESSAGE_TYPES = {
    "FrameMessage": "frame",
    "MeshMessage": "mesh",
    "BoxMessage": "mesh",
    "PointCloudMessage": "point_cloud",
    "LineSegmentsMessage": "line_segments",
    "IcosphereMessage": "icosphere",
    "CylinderMessage": "cylinder",
}


class UnsupportedViserMessageError(RuntimeError):
    pass


@dataclass
class _NormalizerState:
    fps: float
    ordered_node_ids: list[str] = field(default_factory=list)
    nodes: dict[str, NodeManifest] = field(default_factory=dict)
    active_by_path: dict[str, str] = field(default_factory=dict)
    epoch_count_by_path: dict[str, int] = field(default_factory=dict)


MessageHandler = Callable[[_NormalizerState, int, float, dict[str, Any]], None]
CreateDecoder = Callable[[dict[str, Any]], tuple[GeometryData | None, MaterialStyle]]
UpdateDecoder = Callable[[NodeManifest, dict[str, Any], float], GeometryData]


def normalize_recording(recording: RecordingPayload) -> RecordingManifest:
    ordered_times = _ordered_times(recording.messages)
    fps = _infer_fps(ordered_times)
    frame_count = _infer_frame_count(
        ordered_times, duration_seconds=recording.duration_seconds, fps=fps
    )
    duration_seconds = max(float(recording.duration_seconds), 0.0)
    state = _NormalizerState(fps=fps)
    _create_epoch(
        state,
        name="",
        kind="root",
        frame=1,
        time_value=0.0,
        implicit=False,
        geometry=None,
        style=MaterialStyle(),
        initialize_visibility=False,
    )

    for time_value, message in recording.messages:
        frame = _frame_for_time(state.fps, time_value)
        _apply_message(state, frame=frame, time_value=time_value, message=message)

    return RecordingManifest(
        schema_version=2,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        source_viser_version=recording.viser_version,
        nodes=[state.nodes[node_id] for node_id in state.ordered_node_ids],
    )


def _ordered_times(messages: list[tuple[float, dict[str, Any]]]) -> list[float]:
    ordered: list[float] = []
    previous_time: float | None = None
    for time_value, _message in messages:
        if time_value < -1e-9:
            raise UnsupportedViserMessageError(
                f"Negative message time {time_value:.6f} is not supported."
            )
        if previous_time is not None and time_value + 1e-9 < previous_time:
            raise UnsupportedViserMessageError(
                "Message times must be non-decreasing in the recording."
            )
        if not ordered or abs(time_value - ordered[-1]) > 1e-9:
            ordered.append(float(time_value))
        previous_time = float(time_value)
    return ordered


def _apply_message(
    state: _NormalizerState,
    *,
    frame: int,
    time_value: float,
    message: dict[str, Any],
) -> None:
    message_type = message.get("type")
    if not isinstance(message_type, str):
        raise UnsupportedViserMessageError(
            f"Message at t={time_value:.6f} is missing a string type."
        )
    if message_type not in SUPPORTED_MESSAGE_TYPES:
        raise UnsupportedViserMessageError(
            f"Unsupported message type {message_type!r} at t={time_value:.6f}."
        )
    if message_type in CREATE_MESSAGE_TYPES:
        _handle_create_message(
            state, frame=frame, time_value=time_value, message=message
        )
        return
    MESSAGE_HANDLERS[message_type](state, frame, time_value, message)


def _handle_create_message(
    state: _NormalizerState,
    *,
    frame: int,
    time_value: float,
    message: dict[str, Any],
) -> None:
    message_type = cast(str, message["type"])
    name = _require_name(message, message_type=message_type, time_value=time_value)
    props = message.get("props")
    if not isinstance(props, dict):
        raise UnsupportedViserMessageError(
            f"{message_type} at t={time_value:.6f} is missing props."
        )

    kind = CREATE_MESSAGE_TYPES[message_type]
    geometry, style = CREATE_DECODERS[message_type](cast(dict[str, Any], props))
    _coerce_epoch_for_create(
        state,
        name=name,
        kind=kind,
        frame=frame,
        time_value=time_value,
        geometry=geometry,
        style=style,
    )


def _coerce_epoch_for_create(
    state: _NormalizerState,
    *,
    name: str,
    kind: str,
    frame: int,
    time_value: float,
    geometry: GeometryData | None,
    style: MaterialStyle,
) -> NodeManifest:
    active_id = state.active_by_path.get(name)
    if active_id is not None:
        active = state.nodes[active_id]
        if (
            active.kind == "group"
            and active.implicit
            and active.geometry is None
            and active.create_frame == frame
        ):
            active.kind = kind
            active.implicit = False
            active.geometry = geometry
            active.style = style
            return active
        _deactivate_subtree(state, name, frame=frame, time_value=time_value)
    return _create_epoch(
        state,
        name=name,
        kind=kind,
        frame=frame,
        time_value=time_value,
        implicit=False,
        geometry=geometry,
        style=style,
    )


def _create_epoch(
    state: _NormalizerState,
    *,
    name: str,
    kind: str,
    frame: int,
    time_value: float,
    implicit: bool,
    geometry: GeometryData | None,
    style: MaterialStyle,
    initialize_visibility: bool = True,
) -> NodeManifest:
    canonical_name = _canonical_name(name)
    parent_name = _parent_path(canonical_name)
    parent_id = None
    if parent_name is not None:
        parent = _ensure_path_epoch(
            state, parent_name, frame=frame, time_value=time_value
        )
        parent_id = parent.node_id

    epoch_index = state.epoch_count_by_path.get(canonical_name, 0)
    node_id = f"{canonical_name}#{epoch_index}"
    state.epoch_count_by_path[canonical_name] = epoch_index + 1

    manifest = NodeManifest(
        node_id=node_id,
        name=canonical_name,
        kind=kind,
        parent_id=parent_id,
        parent_name=parent_name,
        create_frame=frame,
        create_time_seconds=time_value,
        destroy_frame=None,
        destroy_time_seconds=None,
        implicit=implicit,
        geometry=geometry,
        style=style,
        transform_keyframes=[],
        visibility_keyframes=[],
        geometry_keyframes=[],
    )
    state.nodes[node_id] = manifest
    state.ordered_node_ids.append(node_id)
    state.active_by_path[canonical_name] = node_id

    if initialize_visibility and kind != "root":
        if frame > 1:
            _append_visibility_keyframe(
                manifest, frame=1, time_value=0.0, visible=False
            )
        _append_visibility_keyframe(
            manifest, frame=frame, time_value=time_value, visible=True
        )
    return manifest


def _ensure_path_epoch(
    state: _NormalizerState,
    name: str,
    *,
    frame: int,
    time_value: float,
) -> NodeManifest:
    canonical_name = _canonical_name(name)
    active_id = state.active_by_path.get(canonical_name)
    if active_id is not None:
        return state.nodes[active_id]
    if canonical_name == "":
        return state.nodes[ROOT_NODE_ID]
    return _create_epoch(
        state,
        name=canonical_name,
        kind="group",
        frame=frame,
        time_value=time_value,
        implicit=True,
        geometry=None,
        style=MaterialStyle(),
    )


def _deactivate_subtree(
    state: _NormalizerState,
    name: str,
    *,
    frame: int,
    time_value: float,
) -> None:
    canonical_name = _canonical_name(name)
    prefix = f"{canonical_name}/"
    for path, node_id in list(state.active_by_path.items()):
        if path != canonical_name and not path.startswith(prefix):
            continue
        manifest = state.nodes[node_id]
        manifest.destroy_frame = frame
        manifest.destroy_time_seconds = time_value
        if manifest.kind != "root":
            _append_visibility_keyframe(
                manifest,
                frame=frame,
                time_value=time_value,
                visible=False,
            )
        del state.active_by_path[path]


def _handle_position_message(
    state: _NormalizerState,
    frame: int,
    time_value: float,
    message: dict[str, Any],
) -> None:
    manifest = _target_node(
        state,
        frame=frame,
        time_value=time_value,
        message=message,
        message_type="SetPositionMessage",
        create_implicit=True,
    )
    position = _vector3(
        message.get("position"), field="position", message_type="SetPositionMessage"
    )
    _append_transform_keyframe(
        manifest, frame=frame, time_value=time_value, position=position
    )


def _handle_orientation_message(
    state: _NormalizerState,
    frame: int,
    time_value: float,
    message: dict[str, Any],
) -> None:
    manifest = _target_node(
        state,
        frame=frame,
        time_value=time_value,
        message=message,
        message_type="SetOrientationMessage",
        create_implicit=True,
    )
    rotation = _vector4(
        message.get("wxyz"), field="wxyz", message_type="SetOrientationMessage"
    )
    _append_transform_keyframe(
        manifest, frame=frame, time_value=time_value, rotation_wxyz=rotation
    )


def _handle_visibility_message(
    state: _NormalizerState,
    frame: int,
    time_value: float,
    message: dict[str, Any],
) -> None:
    manifest = _target_node(
        state,
        frame=frame,
        time_value=time_value,
        message=message,
        message_type="SetSceneNodeVisibilityMessage",
        create_implicit=True,
    )
    visible = message.get("visible")
    if not isinstance(visible, bool):
        raise UnsupportedViserMessageError(
            f"SetSceneNodeVisibilityMessage at t={time_value:.6f} "
            f"has invalid visible={visible!r}."
        )
    _append_visibility_keyframe(
        manifest, frame=frame, time_value=time_value, visible=visible
    )


def _handle_remove_message(
    state: _NormalizerState,
    frame: int,
    time_value: float,
    message: dict[str, Any],
) -> None:
    name = _require_name(
        message,
        message_type="RemoveSceneNodeMessage",
        time_value=time_value,
        allow_empty=True,
    )
    if _canonical_name(name) == "":
        return
    _deactivate_subtree(state, name, frame=frame, time_value=time_value)


def _handle_scene_update_message(
    state: _NormalizerState,
    frame: int,
    time_value: float,
    message: dict[str, Any],
) -> None:
    manifest = _target_node(
        state,
        frame=frame,
        time_value=time_value,
        message=message,
        message_type="SceneNodeUpdateMessage",
        create_implicit=False,
    )
    updates = message.get("updates")
    if not isinstance(updates, dict):
        raise UnsupportedViserMessageError(
            f"SceneNodeUpdateMessage at t={time_value:.6f} is missing updates."
        )
    geometry = UPDATE_DECODERS[manifest.kind](
        manifest, cast(dict[str, Any], updates), time_value
    )
    _append_geometry_keyframe(
        manifest, frame=frame, time_value=time_value, geometry=geometry
    )


def _append_transform_keyframe(
    manifest: NodeManifest,
    *,
    frame: int,
    time_value: float,
    position: list[float] | None = None,
    rotation_wxyz: list[float] | None = None,
) -> None:
    keyframes = manifest.transform_keyframes
    if keyframes and keyframes[-1].frame == frame:
        if position is not None:
            keyframes[-1].position = position
        if rotation_wxyz is not None:
            keyframes[-1].rotation_wxyz = rotation_wxyz
        return
    keyframes.append(
        TransformKeyframe(
            frame=frame,
            time_seconds=time_value,
            position=position,
            rotation_wxyz=rotation_wxyz,
        )
    )


def _append_visibility_keyframe(
    manifest: NodeManifest, *, frame: int, time_value: float, visible: bool
) -> None:
    keyframes = manifest.visibility_keyframes
    if keyframes and keyframes[-1].frame == frame:
        keyframes[-1].visible = visible
        keyframes[-1].time_seconds = time_value
        return
    if keyframes and keyframes[-1].visible == visible:
        return
    keyframes.append(
        VisibilityKeyframe(frame=frame, time_seconds=time_value, visible=visible)
    )


def _append_geometry_keyframe(
    manifest: NodeManifest,
    *,
    frame: int,
    time_value: float,
    geometry: GeometryData,
) -> None:
    if frame == manifest.create_frame:
        manifest.geometry = geometry
        return
    keyframes = manifest.geometry_keyframes
    if keyframes and keyframes[-1].frame == frame:
        keyframes[-1].geometry = geometry
        keyframes[-1].time_seconds = time_value
        return
    keyframes.append(
        GeometryKeyframe(frame=frame, time_seconds=time_value, geometry=geometry)
    )


def _target_node(
    state: _NormalizerState,
    *,
    frame: int,
    time_value: float,
    message: dict[str, Any],
    message_type: str,
    create_implicit: bool,
) -> NodeManifest:
    name = _require_name(
        message,
        message_type=message_type,
        time_value=time_value,
        allow_empty=True,
    )
    canonical_name = _canonical_name(name)
    message_prefix = f"{message_type} at t={time_value:.6f}"
    active_id = state.active_by_path.get(canonical_name)
    if active_id is not None:
        return state.nodes[active_id]
    if create_implicit:
        return _ensure_path_epoch(
            state, canonical_name, frame=frame, time_value=time_value
        )
    raise UnsupportedViserMessageError(
        f"{message_prefix} references unknown node {canonical_name!r}."
    )


def _ignore_runtime_message(
    _state: _NormalizerState,
    _frame: int,
    _time_value: float,
    _message: dict[str, Any],
) -> None:
    return


def _ignore_background_message(
    _state: _NormalizerState,
    _frame: int,
    time_value: float,
    message: dict[str, Any],
) -> None:
    if message.get("rgb_data") is not None or message.get("depth_data") is not None:
        raise UnsupportedViserMessageError(
            f"Unsupported non-empty BackgroundImageMessage at t={time_value:.6f}."
        )


def _ignore_message(
    _state: _NormalizerState,
    _frame: int,
    _time_value: float,
    _message: dict[str, Any],
) -> None:
    return


def _decode_frame_payload(
    props: dict[str, Any],
) -> tuple[GeometryData | None, MaterialStyle]:
    geometry = FrameGeometry(
        show_axes=_require_bool(props, "show_axes"),
        axes_length=_require_float(props, "axes_length"),
        axes_radius=_require_float(props, "axes_radius"),
        origin_radius=_require_float(props, "origin_radius"),
        origin_color=cast(
            ColorValue, _decode_color(props.get("origin_color"), item_count=None)
        ),
    )
    return geometry, MaterialStyle()


def _decode_mesh_payload(
    props: dict[str, Any],
) -> tuple[GeometryData | None, MaterialStyle]:
    geometry = MeshGeometry(
        vertices=_decode_vertices(props.get("vertices")),
        faces=_decode_faces(props.get("faces")),
    )
    style = _decode_surface_style(props, include_scale=True)
    return geometry, style


def _decode_box_payload(
    props: dict[str, Any],
) -> tuple[GeometryData | None, MaterialStyle]:
    dimensions = _vector3(
        props.get("dimensions"),
        field="dimensions",
        message_type="BoxMessage",
    )
    half_x, half_y, half_z = (dimension / 2.0 for dimension in dimensions)
    geometry = MeshGeometry(
        vertices=[
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, half_y, -half_z],
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [half_x, half_y, half_z],
            [-half_x, half_y, half_z],
        ],
        faces=[
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
    )
    style = _decode_surface_style(props, include_scale=True)
    return geometry, style


def _decode_point_cloud_payload(
    props: dict[str, Any],
) -> tuple[GeometryData | None, MaterialStyle]:
    precision = _require_str(props, "precision")
    geometry = PointCloudGeometry(
        points=_decode_point_cloud_points(props.get("points"), precision=precision),
        precision=precision,
    )
    point_count = len(geometry.points)
    style = MaterialStyle(
        color=cast(
            ColorValue,
            _decode_color(props.get("colors"), item_count=point_count),
        ),
        point_size=_require_float(props, "point_size"),
        point_shape=_require_str(props, "point_shape"),
    )
    return geometry, style


def _decode_line_segments_payload(
    props: dict[str, Any],
) -> tuple[GeometryData | None, MaterialStyle]:
    geometry = LineSegmentsGeometry(points=_decode_line_points(props.get("points")))
    segment_count = len(geometry.points)
    style = MaterialStyle(
        color=cast(
            ColorValue,
            _decode_color(props.get("colors"), item_count=segment_count, item_width=2),
        ),
        line_width=_require_float(props, "line_width"),
    )
    return geometry, style


def _decode_icosphere_payload(
    props: dict[str, Any],
) -> tuple[GeometryData | None, MaterialStyle]:
    geometry = IcosphereGeometry(
        radius=_require_float(props, "radius"),
        subdivisions=_require_int(props, "subdivisions"),
    )
    style = _decode_surface_style(props)
    return geometry, style


def _decode_cylinder_payload(
    props: dict[str, Any],
) -> tuple[GeometryData | None, MaterialStyle]:
    geometry = CylinderGeometry(
        radius=_require_float(props, "radius"),
        height=_require_float(props, "height"),
        radial_segments=_require_int(props, "radial_segments"),
    )
    style = _decode_surface_style(props)
    return geometry, style


def _decode_mesh_update(
    manifest: NodeManifest, updates: dict[str, Any], time_value: float
) -> GeometryData:
    geometry = _require_geometry(manifest, MeshGeometry)
    return MeshGeometry(
        vertices=_decode_fixed_vertices(
            _decode_vertices(updates["vertices"]),
            expected_count=len(geometry.vertices),
            time_value=time_value,
            node_name=manifest.name,
            label="Mesh vertex",
        ),
        faces=[face[:] for face in geometry.faces],
    )


def _decode_point_cloud_update(
    manifest: NodeManifest, updates: dict[str, Any], time_value: float
) -> GeometryData:
    geometry = _require_geometry(manifest, PointCloudGeometry)
    return PointCloudGeometry(
        points=_decode_fixed_vertices(
            _decode_point_cloud_points(updates["points"], precision=geometry.precision),
            expected_count=len(geometry.points),
            time_value=time_value,
            node_name=manifest.name,
            label="Point",
        ),
        precision=geometry.precision,
    )


def _decode_line_segments_update(
    manifest: NodeManifest, updates: dict[str, Any], time_value: float
) -> GeometryData:
    geometry = _require_geometry(manifest, LineSegmentsGeometry)
    points = _decode_line_points(updates["points"])
    if len(points) != len(geometry.points):
        raise UnsupportedViserMessageError(
            f"Line segment count changed at t={time_value:.6f} for {manifest.name!r}."
        )
    return LineSegmentsGeometry(points=points)


def _decode_fixed_vertices(
    values: list[list[float]],
    *,
    expected_count: int,
    time_value: float,
    node_name: str,
    label: str,
) -> list[list[float]]:
    if len(values) != expected_count:
        raise UnsupportedViserMessageError(
            f"{label} count changed at t={time_value:.6f} for {node_name!r}."
        )
    return values


def _decode_vertices(value: Any) -> list[list[float]]:
    return cast(list[list[float]], _decode_array(value, dtype=np.float32, dims=3))


def _decode_faces(value: Any) -> list[list[int]]:
    return cast(
        list[list[int]],
        _decode_array(value, dtype=np.uint32, dims=3, cast_int=True),
    )


def _decode_point_cloud_points(value: Any, *, precision: str) -> list[list[float]]:
    if precision == "float16":
        dtype = np.float16
    elif precision == "float32":
        dtype = np.float32
    else:
        raise UnsupportedViserMessageError(
            f"Unsupported point cloud precision {precision!r}."
        )
    return cast(list[list[float]], _decode_array(value, dtype=dtype, dims=3))


def _decode_line_points(value: Any) -> list[list[list[float]]]:
    data = _bytes_or_array(value, dtype=np.float32)
    if data.ndim == 1:
        segment_width = 6
        segment_layout = f"{segment_width} values per segment"
        value_count = data.size
        if data.size % segment_width != 0:
            raise UnsupportedViserMessageError(
                f"Expected line segment buffer with {segment_layout}, "
                f"got {value_count}."
            )
        data = data.reshape(-1, 2, 3)
    if data.ndim != 3 or data.shape[1:] != (2, 3):
        raise UnsupportedViserMessageError(
            f"Expected line segment points with shape (N, 2, 3), got {data.shape}."
        )
    return data.astype(np.float32).tolist()


def _decode_scale(value: Any) -> float | list[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [float(component) for component in value]
    raise UnsupportedViserMessageError(f"Unsupported mesh scale {value!r}.")


def _decode_color(
    value: Any,
    *,
    item_count: int | None,
    item_width: int = 1,
) -> ColorValue:
    if isinstance(value, (list, tuple)):
        components = [int(component) for component in value]
        if len(components) not in {3, 4}:
            raise UnsupportedViserMessageError(f"Unsupported color value {value!r}.")
        return components
    if not isinstance(value, (bytes, bytearray)):
        raise UnsupportedViserMessageError(f"Unsupported color value {value!r}.")
    data = np.frombuffer(value, dtype=np.uint8)
    if data.size in {3, 4}:
        return cast(ColorValue, data.astype(np.uint8).tolist())
    if item_count is None:
        raise UnsupportedViserMessageError(
            f"Unsupported color buffer length {data.size}."
        )
    for channels in (3, 4):
        if data.size == item_count * channels:
            return cast(
                ColorValue,
                data.reshape(item_count, channels).astype(np.uint8).tolist(),
            )
        if item_width > 1 and data.size == item_count * item_width * channels:
            return cast(
                ColorValue,
                data.reshape(item_count, item_width, channels)
                .astype(np.uint8)
                .tolist(),
            )
    raise UnsupportedViserMessageError(
        f"Unsupported color buffer length {data.size} for item count {item_count}."
    )


def _decode_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    dims: int,
    cast_int: bool = False,
) -> list[list[float]] | list[list[int]]:
    data = _bytes_or_array(value, dtype=dtype)
    if data.ndim == 1:
        value_count = data.size
        row_shape = f"{dims}-wide rows"
        if data.size % dims != 0:
            raise UnsupportedViserMessageError(
                f"Expected a flat buffer divisible into {row_shape}, got {value_count}."
            )
        data = data.reshape(-1, dims)
    if data.ndim != 2 or data.shape[1] != dims:
        raise UnsupportedViserMessageError(
            f"Expected an array with shape (N, {dims}), got {data.shape}."
        )
    if cast_int:
        return data.astype(np.int64).tolist()
    return data.astype(np.float32).tolist()


def _bytes_or_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic],
) -> np.ndarray[Any, Any]:
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(value, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _require_name(
    message: dict[str, Any],
    *,
    message_type: str,
    time_value: float,
    allow_empty: bool = False,
) -> str:
    name = message.get("name")
    if not isinstance(name, str):
        raise UnsupportedViserMessageError(
            f"{message_type} at t={time_value:.6f} is missing a valid name."
        )
    canonical_name = _canonical_name(name)
    if not allow_empty and canonical_name == "":
        raise UnsupportedViserMessageError(
            f"{message_type} at t={time_value:.6f} is missing a valid name."
        )
    return canonical_name


def _canonical_name(name: str) -> str:
    return "" if name == "/" else name


def _parent_path(name: str) -> str | None:
    canonical_name = _canonical_name(name)
    if canonical_name == "":
        return None
    parent, _, _child = canonical_name.rpartition("/")
    return parent or ""


def _infer_fps(times: list[float]) -> float:
    deltas = [
        curr - prev
        for prev, curr in zip(times, times[1:], strict=False)
        if curr - prev > 1e-9
    ]
    if not deltas:
        return 24.0
    return float(1.0 / median(deltas))


def _infer_frame_count(
    times: list[float], *, duration_seconds: float, fps: float
) -> int:
    last_event_frame = 1
    if times:
        last_event_frame = _frame_for_time(fps, times[-1])
    duration_frame = max(int(round(max(duration_seconds, 0.0) * fps)) + 1, 1)
    return max(last_event_frame, duration_frame, 1)


def _frame_for_time(fps: float, time_value: float) -> int:
    return max(int(round(time_value * fps)) + 1, 1)


def _decode_surface_style(
    props: dict[str, Any],
    *,
    include_scale: bool = False,
) -> MaterialStyle:
    scale = _decode_scale(props.get("scale")) if include_scale else None
    return MaterialStyle(
        color=cast(ColorValue, _decode_color(props.get("color"), item_count=None)),
        scale=scale,
        wireframe=_require_bool(props, "wireframe"),
        opacity=_optional_float(props.get("opacity")),
        flat_shading=_optional_bool(props.get("flat_shading")),
        side=_optional_side(props.get("side")),
        material=_optional_material(props.get("material")),
        cast_shadow=_optional_bool(props.get("cast_shadow")),
        receive_shadow=_optional_bool(props.get("receive_shadow")),
    )


def _require_bool(props: dict[str, Any], key: str) -> bool:
    value = props.get(key)
    if not isinstance(value, bool):
        raise UnsupportedViserMessageError(f"Expected bool {key!r}, got {value!r}.")
    return value


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise UnsupportedViserMessageError(f"Expected optional bool, got {value!r}.")
    return value


def _require_float(props: dict[str, Any], key: str) -> float:
    value = props.get(key)
    if not isinstance(value, (int, float)):
        raise UnsupportedViserMessageError(f"Expected float {key!r}, got {value!r}.")
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise UnsupportedViserMessageError(f"Expected optional float, got {value!r}.")
    return float(value)


def _require_int(props: dict[str, Any], key: str) -> int:
    value = props.get(key)
    if not isinstance(value, int):
        raise UnsupportedViserMessageError(f"Expected int {key!r}, got {value!r}.")
    return value


def _require_str(props: dict[str, Any], key: str) -> str:
    value = props.get(key)
    if not isinstance(value, str):
        raise UnsupportedViserMessageError(f"Expected str {key!r}, got {value!r}.")
    return value


def _optional_side(value: Any) -> str | None:
    if value is None:
        return None
    if value not in {"front", "double"}:
        raise UnsupportedViserMessageError(
            f"Unsupported material side {value!r}; expected 'front' or 'double'."
        )
    return cast(str, value)


def _optional_material(value: Any) -> str | None:
    if value is None:
        return None
    if value != "standard":
        raise UnsupportedViserMessageError(
            f"Unsupported material type {value!r}; expected 'standard'."
        )
    return cast(str, value)


def _vector3(value: Any, *, field: str, message_type: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise UnsupportedViserMessageError(
            f"{message_type} has invalid {field}={value!r}; expected length 3."
        )
    return [float(component) for component in value]


def _vector4(value: Any, *, field: str, message_type: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise UnsupportedViserMessageError(
            f"{message_type} has invalid {field}={value!r}; expected length 4."
        )
    return [float(component) for component in value]


def _require_geometry(manifest: NodeManifest, expected_type: type[Any]) -> GeometryData:
    geometry = _current_geometry(manifest)
    if isinstance(geometry, expected_type):
        return geometry
    raise UnsupportedViserMessageError(
        f"{manifest.kind!r} does not support this geometry update."
    )


def _current_geometry(manifest: NodeManifest) -> GeometryData | None:
    if manifest.geometry_keyframes:
        return manifest.geometry_keyframes[-1].geometry
    return manifest.geometry


CREATE_DECODERS: dict[str, CreateDecoder] = {
    "FrameMessage": _decode_frame_payload,
    "MeshMessage": _decode_mesh_payload,
    "BoxMessage": _decode_box_payload,
    "PointCloudMessage": _decode_point_cloud_payload,
    "LineSegmentsMessage": _decode_line_segments_payload,
    "IcosphereMessage": _decode_icosphere_payload,
    "CylinderMessage": _decode_cylinder_payload,
}

UPDATE_DECODERS: dict[str, UpdateDecoder] = {
    "mesh": _decode_mesh_update,
    "point_cloud": _decode_point_cloud_update,
    "line_segments": _decode_line_segments_update,
}

MESSAGE_HANDLERS: dict[str, MessageHandler] = {
    "RunJavascriptMessage": _ignore_runtime_message,
    "BackgroundImageMessage": _ignore_background_message,
    "SetGuiPanelLabelMessage": _ignore_message,
    "SetPositionMessage": _handle_position_message,
    "SetOrientationMessage": _handle_orientation_message,
    "SetSceneNodeVisibilityMessage": _handle_visibility_message,
    "RemoveSceneNodeMessage": _handle_remove_message,
    "SceneNodeUpdateMessage": _handle_scene_update_message,
}
