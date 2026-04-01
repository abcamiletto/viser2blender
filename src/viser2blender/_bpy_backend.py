from __future__ import annotations

import importlib
import importlib.util
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, cast

from ._types import (
    ColorValue,
    CylinderGeometry,
    FrameGeometry,
    GeometryData,
    IcosphereGeometry,
    LineSegmentsGeometry,
    MaterialStyle,
    MeshGeometry,
    NodeManifest,
    PointCloudGeometry,
    RecordingManifest,
)


MIN_BPY_VERSION = (5, 1, 0)


def convert_manifest(scene_manifest: RecordingManifest, output_path: Path) -> None:
    bpy = _import_bpy()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    fps_numerator, fps_denominator = _fps_fraction(scene_manifest.fps)
    scene.render.fps = fps_numerator
    scene.render.fps_base = float(fps_denominator)
    scene.frame_start = 1
    scene.frame_end = int(scene_manifest.frame_count)

    objects = {
        node.node_id: _create_node_object(bpy, node) for node in scene_manifest.nodes
    }

    for node in scene_manifest.nodes:
        if node.parent_id is not None:
            objects[node.node_id].parent = objects[node.parent_id]

    for node in scene_manifest.nodes:
        obj = objects[node.node_id]
        _apply_transform_keyframes(obj, node)
        _apply_visibility_keyframes(obj, node)
        _apply_geometry_keyframes(obj, node)

    _set_constant_interpolation(bpy)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), compress=False)


def _import_bpy() -> Any:
    if importlib.util.find_spec("bpy") is None:
        raise RuntimeError("viser2blender requires Blender bpy>=5.1.0.")
    bpy = importlib.import_module("bpy")
    version = tuple(cast(tuple[int, int, int], bpy.app.version))
    if version < MIN_BPY_VERSION:
        found = ".".join(str(part) for part in version)
        raise RuntimeError(
            f"viser2blender requires Blender bpy>=5.1.0. Found bpy {found}."
        )
    return bpy


def _fps_fraction(fps: float) -> tuple[int, int]:
    ratio = Fraction(fps).limit_denominator(1001)
    return max(ratio.numerator, 1), max(ratio.denominator, 1)


def _safe_name(name: str) -> str:
    stripped = name.strip("/")
    return stripped.replace("/", "__") or "root"


def _blender_name(node: NodeManifest) -> str:
    base = _safe_name(node.name)
    epoch = node.node_id.rsplit("#", 1)[1]
    return base if epoch == "0" else f"{base}__epoch{epoch}"


def _first_color_leaf(color: ColorValue) -> list[int]:
    sample: Any = color
    while sample and isinstance(sample[0], list):
        sample = sample[0]
    return cast(list[int], sample)


def _ensure_material(bpy: Any, name: str, style: MaterialStyle) -> Any | None:
    if style.color is None:
        return None
    rgba = _normalized_rgba(_first_color_leaf(style.color), style.opacity)
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True

    principled = material.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Alpha"].default_value = rgba[3]
    if rgba[3] < 1.0:
        material.blend_method = "HASHED"
    material.use_backface_culling = style.side == "front"
    return material


def _create_group_object(bpy: Any, _node: NodeManifest) -> Any:
    return _add_empty_object(bpy, "PLAIN_AXES")


def _create_frame_object(bpy: Any, node: NodeManifest) -> Any:
    geometry = cast(FrameGeometry, node.geometry)
    obj = _add_empty_object(bpy, "ARROWS" if geometry.show_axes else "PLAIN_AXES")
    obj.empty_display_size = float(geometry.axes_length)
    obj["viser_axes_radius"] = float(geometry.axes_radius)
    obj["viser_origin_radius"] = float(geometry.origin_radius)
    obj["viser_show_axes"] = bool(geometry.show_axes)
    return obj


def _create_mesh_object(
    bpy: Any,
    name: str,
    vertices: list[list[float]],
    faces: list[list[int]],
    edges: list[tuple[int, int]] | None = None,
) -> Any:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges or [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _create_mesh_node(bpy: Any, node: NodeManifest) -> Any:
    geometry = cast(MeshGeometry, node.geometry)
    return _create_mesh_object(
        bpy,
        _blender_name(node),
        geometry.vertices,
        geometry.faces,
    )


def _create_point_cloud_node(bpy: Any, node: NodeManifest) -> Any:
    geometry = cast(PointCloudGeometry, node.geometry)
    obj = _create_mesh_object(bpy, _blender_name(node), geometry.points, [])
    obj["viser_point_precision"] = geometry.precision
    return obj


def _create_line_segments_node(bpy: Any, node: NodeManifest) -> Any:
    geometry = cast(LineSegmentsGeometry, node.geometry)
    vertices = [endpoint for segment in geometry.points for endpoint in segment]
    edges = [(index, index + 1) for index in range(0, len(vertices), 2)]
    return _create_mesh_object(bpy, _blender_name(node), vertices, [], edges)


def _create_icosphere_node(bpy: Any, node: NodeManifest) -> Any:
    geometry = cast(IcosphereGeometry, node.geometry)
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=int(geometry.subdivisions),
        radius=float(geometry.radius),
    )
    return _active_object(bpy)


def _create_cylinder_node(bpy: Any, node: NodeManifest) -> Any:
    geometry = cast(CylinderGeometry, node.geometry)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=int(geometry.radial_segments),
        radius=float(geometry.radius),
        depth=float(geometry.height),
    )
    return _active_object(bpy)


def _create_node_object(bpy: Any, node: NodeManifest) -> Any:
    obj = OBJECT_BUILDERS[node.kind](bpy, node)
    obj.name = _blender_name(node)
    obj["viser_node_id"] = node.node_id
    obj["viser_name"] = node.name
    obj["viser_kind"] = node.kind
    obj["viser_implicit"] = node.implicit
    obj.rotation_mode = "QUATERNION"

    _apply_style(bpy, obj, node)
    return obj


def _apply_style(bpy: Any, obj: Any, node: NodeManifest) -> None:
    style = node.style
    if style.scale is not None:
        obj.scale = _scale_vector(style.scale)
    if style.wireframe:
        obj.display_type = "WIRE"
        if obj.type == "MESH":
            obj.show_wire = True
            obj.show_all_edges = True
    if style.cast_shadow is not None and hasattr(obj, "visible_shadow"):
        obj.visible_shadow = bool(style.cast_shadow)
    if style.receive_shadow is not None:
        obj["viser_receive_shadow"] = bool(style.receive_shadow)
    if style.point_size is not None:
        obj["viser_point_size"] = float(style.point_size)
    if style.point_shape is not None:
        obj["viser_point_shape"] = str(style.point_shape)
    if style.line_width is not None:
        obj["viser_line_width"] = float(style.line_width)
    if style.side is not None:
        obj["viser_side"] = style.side
    if style.material is not None:
        obj["viser_material"] = style.material

    material = _ensure_material(bpy, f"{_blender_name(node)}_material", style)
    if material is not None and obj.type == "MESH":
        obj.data.materials.append(material)
    elif style.color is not None and hasattr(obj, "color"):
        rgba = _normalized_rgba(_first_color_leaf(style.color), style.opacity)
        obj.color = rgba

    if obj.type == "MESH" and style.flat_shading is not None:
        for polygon in obj.data.polygons:
            polygon.use_smooth = not style.flat_shading


def _normalized_rgba(color: list[int], opacity: float | None) -> list[float]:
    rgba = [float(channel) / 255.0 for channel in color]
    if len(rgba) == 3:
        rgba.append(1.0 if opacity is None else float(opacity))
    elif opacity is not None:
        rgba[3] = float(opacity)
    return rgba


def _scale_vector(scale: float | list[float]) -> tuple[float, float, float]:
    if isinstance(scale, float):
        return (scale, scale, scale)
    return (float(scale[0]), float(scale[1]), float(scale[2]))


def _add_empty_object(bpy: Any, empty_type: str) -> Any:
    bpy.ops.object.empty_add(type=empty_type)
    return _active_object(bpy)


def _active_object(bpy: Any) -> Any:
    obj = bpy.context.active_object
    assert obj is not None
    return obj


def _apply_transform_keyframes(obj: Any, node: NodeManifest) -> None:
    for keyframe in node.transform_keyframes:
        if keyframe.position is not None:
            obj.location = keyframe.position
            obj.keyframe_insert(data_path="location", frame=keyframe.frame)
        if keyframe.rotation_wxyz is not None:
            obj.rotation_quaternion = keyframe.rotation_wxyz
            obj.keyframe_insert(data_path="rotation_quaternion", frame=keyframe.frame)


def _apply_visibility_keyframes(obj: Any, node: NodeManifest) -> None:
    for keyframe in node.visibility_keyframes:
        obj.hide_viewport = not keyframe.visible
        obj.hide_render = not keyframe.visible
        obj.keyframe_insert(data_path="hide_viewport", frame=keyframe.frame)
        obj.keyframe_insert(data_path="hide_render", frame=keyframe.frame)


def _apply_geometry_keyframes(obj: Any, node: NodeManifest) -> None:
    if not node.geometry_keyframes:
        return

    basis = obj.shape_key_add(name="Basis", from_mix=False)
    basis.interpolation = "KEY_LINEAR"
    previous_shape_name: str | None = None

    for keyframe in node.geometry_keyframes:
        shape_key = obj.shape_key_add(
            name=f"frame_{keyframe.frame:04d}", from_mix=False
        )
        shape_key.interpolation = "KEY_LINEAR"
        vertices = _geometry_vertices(keyframe.geometry)
        for vertex, coord in zip(shape_key.data, vertices, strict=True):
            vertex.co = coord

        start_frame = max(node.create_frame, keyframe.frame - 1)
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=start_frame)
        shape_key.value = 1.0
        shape_key.keyframe_insert(data_path="value", frame=keyframe.frame)

        if previous_shape_name is not None:
            previous_shape = obj.data.shape_keys.key_blocks[previous_shape_name]
            previous_shape.value = 0.0
            previous_shape.keyframe_insert(data_path="value", frame=keyframe.frame)

        previous_shape_name = shape_key.name


def _geometry_vertices(geometry: GeometryData) -> list[list[float]]:
    if isinstance(geometry, MeshGeometry):
        return geometry.vertices
    if isinstance(geometry, PointCloudGeometry):
        return geometry.points
    if isinstance(geometry, LineSegmentsGeometry):
        return [endpoint for segment in geometry.points for endpoint in segment]
    raise RuntimeError(
        f"Geometry updates are not supported for {type(geometry).__name__}."
    )


def _set_constant_interpolation(bpy: Any) -> None:
    for action in bpy.data.actions:
        for curve in _action_fcurves(action):
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "CONSTANT"


def _action_fcurves(action: Any) -> Any:
    fcurves = getattr(action, "fcurves", None)
    if fcurves is not None:
        return fcurves

    layered_fcurves: list[Any] = []
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                layered_fcurves.extend(channelbag.fcurves)
    return layered_fcurves


ObjectBuilder = Callable[[Any, NodeManifest], Any]

OBJECT_BUILDERS: dict[str, ObjectBuilder] = {
    "root": _create_group_object,
    "group": _create_group_object,
    "frame": _create_frame_object,
    "mesh": _create_mesh_node,
    "point_cloud": _create_point_cloud_node,
    "line_segments": _create_line_segments_node,
    "icosphere": _create_icosphere_node,
    "cylinder": _create_cylinder_node,
}
