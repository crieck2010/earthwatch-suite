"""Minimal pure-Python PNG codec (8-bit RGB/RGBA, non-interlaced).

The engine core stays stdlib+numpy: this module reads the PNG frames that
survey-flow (or anything else) wrote, and writes the synthetic PNGs used
by the offline demo and tests. Anything fancier (16-bit, interlaced,
palette) raises a clear error — install imageio/matplotlib for those.
"""

from __future__ import annotations

import binascii
import struct
import zlib
from typing import Tuple

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + chunk_type + data
            + struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF))


def write_png(path: str, img: np.ndarray) -> str:
    """Write a ``(H, W, 3)`` or ``(H, W, 4)`` uint8 array as PNG."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        raise ValueError(f"write_png needs uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"write_png needs (H, W, 3|4), got shape {img.shape}")
    h, w, c = img.shape
    color_type = 2 if c == 3 else 6
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + img[row].tobytes() for row in range(h))
    with open(path, "wb") as fh:
        fh.write(_SIGNATURE)
        fh.write(_chunk(b"IHDR", ihdr))
        fh.write(_chunk(b"IDAT", zlib.compress(raw, 6)))
        fh.write(_chunk(b"IEND", b""))
    return path


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path: str) -> np.ndarray:
    """Read a PNG file into a ``(H, W, 3)`` or ``(H, W, 4)`` uint8 array.

    Supports 8-bit non-interlaced truecolor (2), truecolor+alpha (6),
    and grayscale (0, expanded to RGB). All PNG filter types 0-4 are
    handled.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != _SIGNATURE:
        raise ValueError(f"not a PNG file: {path}")
    pos, w = 8, 0
    h, bit_depth, color_type, interlace = 0, 0, 0, 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            w, h, bit_depth, color_type, _comp, _filt, interlace = \
                struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if bit_depth != 8:
        raise ValueError(f"only 8-bit PNG supported, got {bit_depth}-bit: {path}")
    if interlace != 0:
        raise ValueError(f"interlaced PNG not supported: {path}")
    if color_type == 0:
        channels = 1
    elif color_type == 2:
        channels = 3
    elif color_type == 6:
        channels = 4
    else:
        raise ValueError(f"unsupported PNG color type {color_type}: {path}")
    raw = zlib.decompress(bytes(idat))
    stride = w * channels
    img = np.empty((h, stride), dtype=np.uint8)
    prev = bytearray(stride)
    off = 0
    bpp = channels  # bytes per pixel for filter reconstruction (8-bit)
    for y in range(h):
        ftype = raw[off]
        off += 1
        cur = bytearray(raw[off:off + stride])
        off += stride
        recon = bytearray(stride)
        if ftype == 0:  # None
            recon[:] = cur
        elif ftype == 1:  # Sub
            for i in range(stride):
                a = recon[i - bpp] if i >= bpp else 0
                recon[i] = (cur[i] + a) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                recon[i] = (cur[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                a = recon[i - bpp] if i >= bpp else 0
                recon[i] = (cur[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = recon[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                recon[i] = (cur[i] + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter type {ftype}: {path}")
        img[y] = recon
        prev = recon
    img = img.reshape(h, w, channels)
    if channels == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def png_size(path: str) -> Tuple[int, int]:
    """Return ``(width, height)`` from the IHDR without decoding pixels."""
    with open(path, "rb") as fh:
        data = fh.read(33)
    if data[:8] != _SIGNATURE:
        raise ValueError(f"not a PNG file: {path}")
    (length,) = struct.unpack(">I", data[8:12])
    if data[12:16] != b"IHDR" or length != 13:
        raise ValueError(f"malformed PNG header: {path}")
    w, h = struct.unpack(">II", data[16:24])
    return w, h
