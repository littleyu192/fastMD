"""Reproduce the unmodified WBM initial-structure fixture using only the stdlib.

Run ``python tests/data/fetch_wbm_fixture.py`` from the FastMD checkout.
Only the first 3,179 bytes of the published ZIP are requested/downloaded.
Alternatively use ``--archive /path/to/2024-08-04-wbm-initial-atoms.extxyz.zip``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import urllib.request
import zlib


URL = "https://ndownloader.figshare.com/files/48169597"
MEMBER = "wbm-1-12.extxyz"
SHA256 = "6975404b868966a5cbc7a053f7ba12769278b009885f2e8c5df9d36bf61110d2"
PREFIX_BYTES = 3179
MEMBER_OFFSET = 2918
DEFAULT_OUTPUT = Path(__file__).with_name("wbm-1-12.initial.extxyz")


def extract(prefix: bytes) -> bytes:
    """Validate and decompress the selected ZIP local-file record."""
    if len(prefix) != PREFIX_BYTES:
        raise ValueError(f"expected {PREFIX_BYTES} archive bytes, got {len(prefix)}")
    fields = struct.unpack_from("<4s5H3I2H", prefix, MEMBER_OFFSET)
    signature, _, flags, method, _, _, crc, size, raw_size, name_size, extra_size = fields
    if signature != b"PK\x03\x04" or method != 8 or flags & (1 | 8):
        raise ValueError("unexpected ZIP local-file header or compression flags")
    start = MEMBER_OFFSET + 30
    name = prefix[start:start + name_size].decode("utf-8")
    if name != MEMBER:
        raise ValueError(f"expected ZIP member {MEMBER!r}, got {name!r}")
    start += name_size + extra_size
    if start + size != PREFIX_BYTES:
        raise ValueError("unexpected ZIP member length")
    payload = zlib.decompress(prefix[start:start + size], -15)
    if len(payload) != raw_size or zlib.crc32(payload) != crc:
        raise ValueError("ZIP member size or CRC32 mismatch")
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("initial-structure SHA256 mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="use a locally downloaded source ZIP")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.archive is not None:
        with args.archive.open("rb") as source:
            prefix = source.read(PREFIX_BYTES)
    else:
        request = urllib.request.Request(URL, headers={"Range": f"bytes=0-{PREFIX_BYTES - 1}"})
        with urllib.request.urlopen(request, timeout=60) as source:
            # Read only the prefix even if a server ignores the Range header.
            prefix = source.read(PREFIX_BYTES)
    payload = extract(prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"{MEMBER}: {len(payload)} bytes, SHA256={SHA256}")
    print(args.output)


if __name__ == "__main__":
    main()
