from __future__ import annotations

import mmap
import numpy as np
import struct
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO


GGUF_MAGIC = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFTensorType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 30


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass(frozen=True, slots=True)
class GGMLTypeInfo:
    block_size: int
    type_size: int
    physical_dtype: str | None


GGML_TYPE_INFO: dict[GGUFTensorType, GGMLTypeInfo] = {
    GGUFTensorType.F32: GGMLTypeInfo(1, 4, "float32"),
    GGUFTensorType.F16: GGMLTypeInfo(1, 2, "float16"),
    GGUFTensorType.I8: GGMLTypeInfo(1, 1, "int8"),
    GGUFTensorType.I16: GGMLTypeInfo(1, 2, "int16"),
    GGUFTensorType.I32: GGMLTypeInfo(1, 4, "int32"),
    GGUFTensorType.I64: GGMLTypeInfo(1, 8, "int64"),
    GGUFTensorType.BF16: GGMLTypeInfo(1, 2, "bfloat16"),
    GGUFTensorType.Q8_0: GGMLTypeInfo(32, 34, None),
    GGUFTensorType.Q2_K: GGMLTypeInfo(256, 84, None),
    GGUFTensorType.Q3_K: GGMLTypeInfo(256, 110, None),
    GGUFTensorType.Q4_K: GGMLTypeInfo(256, 144, None),
    GGUFTensorType.Q5_K: GGMLTypeInfo(256, 176, None),
    GGUFTensorType.Q6_K: GGMLTypeInfo(256, 210, None),
    GGUFTensorType.Q8_K: GGMLTypeInfo(256, 292, None),
}


@dataclass(frozen=True, slots=True)
class GGUFTensorEntry:
    name: str
    ggml_type: str
    ggml_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    physical_dtype: str
    physical_shape: tuple[int, ...]
    data_offset: int
    nbytes: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GGUFIndex:
    path: Path
    version: int
    alignment: int
    metadata: dict[str, object]
    tensors: dict[str, GGUFTensorEntry]


def load_gguf_index(path: str | Path) -> GGUFIndex:
    resolved = Path(path)
    with resolved.open("rb") as handle:
        magic = _read_u32(handle)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{resolved} is not a GGUF file")
        version = _read_u32(handle)
        if version not in {2, 3}:
            raise ValueError(f"Unsupported GGUF version {version}")
        tensor_count = _read_u64(handle)
        kv_count = _read_u64(handle)

        metadata: dict[str, object] = {"GGUF.version": version}
        for _ in range(kv_count):
            key = _read_string(handle)
            value_type = GGUFValueType(_read_u32(handle))
            metadata[key] = _read_metadata_value(handle, value_type)

        descriptors: list[tuple[str, tuple[int, ...], GGUFTensorType, int]] = []
        for _ in range(tensor_count):
            name = _read_string(handle)
            n_dims = _read_u32(handle)
            dims = tuple(_read_u64(handle) for _ in range(n_dims))
            ggml_type = GGUFTensorType(_read_u32(handle))
            tensor_offset = _read_u64(handle)
            descriptors.append((name, dims, ggml_type, tensor_offset))

        alignment = _extract_alignment(metadata)
        data_start = handle.tell()
        padding = data_start % alignment
        if padding:
            data_start += alignment - padding

    tensors = {
        name: _build_tensor_entry(name, dims, ggml_type, data_start + tensor_offset)
        for name, dims, ggml_type, tensor_offset in descriptors
    }
    return GGUFIndex(
        path=resolved,
        version=version,
        alignment=alignment,
        metadata=metadata,
        tensors=tensors,
    )


def read_tensor_prefix(path: str | Path, entry: GGUFTensorEntry, size: int = 64) -> bytes:
    with Path(path).open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return bytes(mapped[entry.data_offset : entry.data_offset + min(size, entry.nbytes)])
        finally:
            mapped.close()


def read_tensor_bytes(path: str | Path, entry: GGUFTensorEntry, offset: int, size: int) -> bytes:
    if offset < 0 or size <= 0 or offset + size > entry.nbytes:
        raise ValueError(
            f"Invalid tensor byte slice for {entry.name}: offset={offset}, size={size}, nbytes={entry.nbytes}"
        )
    with Path(path).open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            start = entry.data_offset + offset
            return bytes(mapped[start : start + size])
        finally:
            mapped.close()


def dequantize_q4_k_blocks(raw_blocks: np.ndarray) -> np.ndarray:
    if raw_blocks.dtype != np.uint8 or raw_blocks.ndim != 2 or raw_blocks.shape[1] != 144:
        raise ValueError(f"Expected uint8 Q4_K blocks with shape [N, 144], got {raw_blocks.shape} {raw_blocks.dtype}")
    n_blocks = raw_blocks.shape[0]
    d = raw_blocks[:, 0:2].view(np.float16).astype(np.float32)
    dmin = raw_blocks[:, 2:4].view(np.float16).astype(np.float32)
    scales = raw_blocks[:, 4:16]
    qs = raw_blocks[:, 16:]

    scale, minimum = unpack_q4_k_scale_min(scales)
    d = (d * scale.astype(np.float32)).reshape((n_blocks, 8, 1))
    dmin = (dmin * minimum.astype(np.float32)).reshape((n_blocks, 8, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)
    return (d * qs - dmin).reshape((n_blocks, 256)).astype(np.float32, copy=False)


def unpack_q4_k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if scales.dtype != np.uint8 or scales.ndim != 2 or scales.shape[1] != 12:
        raise ValueError(f"Expected uint8 scale bytes with shape [N, 12], got {scales.shape} {scales.dtype}")
    n_blocks = scales.shape[0]
    reshaped = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = np.split(reshaped, 3, axis=-2)
    scale = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    minimum = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return scale.reshape((n_blocks, 8)), minimum.reshape((n_blocks, 8))


def _build_tensor_entry(
    name: str,
    dims: tuple[int, ...],
    ggml_type: GGUFTensorType,
    data_offset: int,
) -> GGUFTensorEntry:
    info = GGML_TYPE_INFO[ggml_type]
    logical_shape = tuple(reversed(dims))
    elements = 1
    for dim in dims:
        elements *= int(dim)
    nbytes = elements * info.type_size // info.block_size
    if info.physical_dtype is not None:
        physical_dtype = info.physical_dtype
        physical_shape = logical_shape
    else:
        byte_shape = (*logical_shape[:-1], logical_shape[-1] // info.block_size * info.type_size)
        if ggml_type is GGUFTensorType.Q4_K:
            physical_dtype = "uint32"
            physical_shape = (*byte_shape[:-1], byte_shape[-1] // 4)
        elif ggml_type in {GGUFTensorType.Q8_0, GGUFTensorType.Q6_K}:
            physical_dtype = "uint16"
            physical_shape = (*byte_shape[:-1], byte_shape[-1] // 2)
        else:
            physical_dtype = "uint8"
            physical_shape = byte_shape
    return GGUFTensorEntry(
        name=name,
        ggml_type=ggml_type.name,
        ggml_shape=dims,
        logical_shape=logical_shape,
        physical_dtype=physical_dtype,
        physical_shape=physical_shape,
        data_offset=data_offset,
        nbytes=nbytes,
    )


def _extract_alignment(metadata: dict[str, object]) -> int:
    value = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    if not isinstance(value, int):
        raise ValueError(f"Invalid GGUF alignment {value!r}")
    return value


def _read_metadata_value(handle: BinaryIO, value_type: GGUFValueType) -> object:
    if value_type is GGUFValueType.UINT8:
        return _read_u8(handle)
    if value_type is GGUFValueType.INT8:
        return _read_i8(handle)
    if value_type is GGUFValueType.UINT16:
        return _read_u16(handle)
    if value_type is GGUFValueType.INT16:
        return _read_i16(handle)
    if value_type is GGUFValueType.UINT32:
        return _read_u32(handle)
    if value_type is GGUFValueType.INT32:
        return _read_i32(handle)
    if value_type is GGUFValueType.FLOAT32:
        return _read_f32(handle)
    if value_type is GGUFValueType.BOOL:
        return bool(_read_u8(handle))
    if value_type is GGUFValueType.STRING:
        return _read_string(handle)
    if value_type is GGUFValueType.ARRAY:
        element_type = GGUFValueType(_read_u32(handle))
        return [_read_metadata_value(handle, element_type) for _ in range(_read_u64(handle))]
    if value_type is GGUFValueType.UINT64:
        return _read_u64(handle)
    if value_type is GGUFValueType.INT64:
        return _read_i64(handle)
    if value_type is GGUFValueType.FLOAT64:
        return _read_f64(handle)
    raise ValueError(f"Unsupported GGUF value type: {value_type}")


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("Truncated GGUF string")
    return raw.decode("utf-8")


def _read_u8(handle: BinaryIO) -> int:
    return _unpack_int("<B", handle)


def _read_i8(handle: BinaryIO) -> int:
    return _unpack_int("<b", handle)


def _read_u16(handle: BinaryIO) -> int:
    return _unpack_int("<H", handle)


def _read_i16(handle: BinaryIO) -> int:
    return _unpack_int("<h", handle)


def _read_u32(handle: BinaryIO) -> int:
    return _unpack_int("<I", handle)


def _read_i32(handle: BinaryIO) -> int:
    return _unpack_int("<i", handle)


def _read_u64(handle: BinaryIO) -> int:
    return _unpack_int("<Q", handle)


def _read_i64(handle: BinaryIO) -> int:
    return _unpack_int("<q", handle)


def _read_f32(handle: BinaryIO) -> float:
    return _unpack_float("<f", handle)


def _read_f64(handle: BinaryIO) -> float:
    return _unpack_float("<d", handle)


def _unpack_int(fmt: str, handle: BinaryIO) -> int:
    value = struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]
    if not isinstance(value, int):
        raise ValueError(f"Expected int for {fmt}")
    return value


def _unpack_float(fmt: str, handle: BinaryIO) -> float:
    value = struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]
    if not isinstance(value, float):
        raise ValueError(f"Expected float for {fmt}")
    return value
