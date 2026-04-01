from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast


JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
ColorValue = list[int] | list[list[int]] | list[list[list[int]]]


@dataclass(slots=True)
class RecordingPayload:
    duration_seconds: float
    viser_version: str
    messages: list[tuple[float, dict[str, Any]]]


@dataclass(slots=True)
class TransformKeyframe:
    frame: int
    time_seconds: float
    position: list[float] | None = None
    rotation_wxyz: list[float] | None = None


@dataclass(slots=True)
class VisibilityKeyframe:
    frame: int
    time_seconds: float
    visible: bool


@dataclass(slots=True)
class MaterialStyle:
    color: ColorValue | None = None
    scale: float | list[float] | None = None
    wireframe: bool | None = None
    opacity: float | None = None
    point_size: float | None = None
    point_shape: str | None = None
    line_width: float | None = None
    flat_shading: bool | None = None
    side: str | None = None
    material: str | None = None
    cast_shadow: bool | None = None
    receive_shadow: bool | None = None


@dataclass(slots=True)
class FrameGeometry:
    show_axes: bool
    axes_length: float
    axes_radius: float
    origin_radius: float
    origin_color: ColorValue


@dataclass(slots=True)
class MeshGeometry:
    vertices: list[list[float]]
    faces: list[list[int]]


@dataclass(slots=True)
class PointCloudGeometry:
    points: list[list[float]]
    precision: str


@dataclass(slots=True)
class LineSegmentsGeometry:
    points: list[list[list[float]]]


@dataclass(slots=True)
class IcosphereGeometry:
    radius: float
    subdivisions: int


@dataclass(slots=True)
class CylinderGeometry:
    radius: float
    height: float
    radial_segments: int


GeometryData = (
    FrameGeometry
    | MeshGeometry
    | PointCloudGeometry
    | LineSegmentsGeometry
    | IcosphereGeometry
    | CylinderGeometry
)


@dataclass(slots=True)
class GeometryKeyframe:
    frame: int
    time_seconds: float
    geometry: GeometryData


@dataclass(slots=True)
class NodeManifest:
    node_id: str
    name: str
    kind: str
    parent_id: str | None
    parent_name: str | None
    create_frame: int
    create_time_seconds: float
    destroy_frame: int | None
    destroy_time_seconds: float | None
    implicit: bool
    geometry: GeometryData | None
    style: MaterialStyle
    transform_keyframes: list[TransformKeyframe]
    visibility_keyframes: list[VisibilityKeyframe]
    geometry_keyframes: list[GeometryKeyframe]


@dataclass(slots=True)
class RecordingManifest:
    schema_version: int
    fps: float
    frame_count: int
    duration_seconds: float
    source_viser_version: str
    nodes: list[NodeManifest]

    def to_jsonable(self) -> dict[str, JSONValue]:
        return cast(dict[str, JSONValue], asdict(self))
