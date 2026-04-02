from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import msgspec
import zstandard

from ._types import RecordingPayload


def load_viser_recording(path: str | Path) -> RecordingPayload:
    input_path = Path(path)
    raw = input_path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{input_path} is too short to be a valid .viser file.")

    packed_size = int.from_bytes(raw[:8], "little")
    packed = zstandard.ZstdDecompressor().decompress(raw[8:])
    if packed_size != len(packed):
        raise ValueError(
            f"{input_path} decompressed to {len(packed)} bytes, expected {packed_size}."
        )

    payload = _decode_payload(packed, input_path)
    duration_seconds = payload.get("durationSeconds")
    viser_version = payload.get("viserVersion")
    messages = payload.get("messages")
    if not isinstance(duration_seconds, (int, float)):
        raise ValueError(f"{input_path} is missing durationSeconds.")
    if not isinstance(viser_version, str):
        raise ValueError(f"{input_path} is missing viserVersion.")
    if not isinstance(messages, list):
        raise ValueError(f"{input_path} is missing messages.")

    parsed_messages: list[tuple[float, dict[str, Any]]] = []
    for entry in messages:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"{input_path} has an invalid message entry: {entry!r}")
        time_value, message = entry
        if not isinstance(time_value, (int, float)) or not isinstance(message, dict):
            raise ValueError(f"{input_path} has an invalid message entry: {entry!r}")
        parsed_messages.append((float(time_value), cast(dict[str, Any], message)))

    return RecordingPayload(
        duration_seconds=float(duration_seconds),
        viser_version=viser_version,
        messages=parsed_messages,
    )


def _decode_payload(packed: bytes, input_path: Path) -> dict[str, Any]:
    msgpack_size = _hybrid_msgpack_size(packed)
    if msgpack_size is None:
        if packed[:1] not in {b"\x83", b"\x84"}:
            raise ValueError(f"{input_path} is not a supported .viser payload.")
        return cast(dict[str, Any], msgspec.msgpack.decode(packed))

    payload = cast(dict[str, Any], msgspec.msgpack.decode(packed[8 : 8 + msgpack_size]))
    buffer_lengths = payload.get("binaryBufferLengths")
    if not isinstance(buffer_lengths, list):
        return payload

    buffer_offsets = _compute_buffer_offsets(
        [int(length) for length in buffer_lengths],
        start_offset=8 + msgpack_size,
    )
    payload = cast(
        dict[str, Any],
        _replace_binary_placeholders(payload, packed, buffer_offsets, buffer_lengths),
    )
    payload.pop("binaryBufferLengths", None)
    return payload


def _hybrid_msgpack_size(packed: bytes) -> int | None:
    if len(packed) < 9:
        return None

    msgpack_size = int.from_bytes(packed[:8], "little")
    if msgpack_size <= 0 or 8 + msgpack_size > len(packed):
        return None
    if packed[8] not in {0x83, 0x84}:
        return None
    return msgpack_size


def _compute_buffer_offsets(
    buffer_lengths: list[int],
    *,
    start_offset: int,
) -> list[int]:
    offsets: list[int] = []
    current_offset = start_offset
    for length in buffer_lengths:
        if length < 0:
            raise ValueError("Binary buffer lengths must be non-negative.")
        current_offset += (8 - (current_offset % 8)) % 8
        offsets.append(current_offset)
        current_offset += length
    return offsets


def _replace_binary_placeholders(
    value: Any,
    packed: bytes,
    buffer_offsets: list[int],
    buffer_lengths: list[int],
) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"__binary_index", "dtype"}:
            index = value["__binary_index"]
            if not isinstance(index, int) or not 0 <= index < len(buffer_offsets):
                raise ValueError(f"Invalid binary buffer index {index!r}.")
            start = buffer_offsets[index]
            end = start + buffer_lengths[index]
            return packed[start:end]
        return {
            key: _replace_binary_placeholders(
                item,
                packed,
                buffer_offsets,
                buffer_lengths,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _replace_binary_placeholders(item, packed, buffer_offsets, buffer_lengths)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _replace_binary_placeholders(item, packed, buffer_offsets, buffer_lengths)
            for item in value
        )

    return value
