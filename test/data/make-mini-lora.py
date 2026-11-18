#!/usr/bin/env python3
"""Write a tiny rank-1 LoRA adapter for models/stories260K.gguf.

The base model is 5 layers x 64 embd, so a rank-1 delta on blk.0.attn_q is
a few hundred bytes.  Regenerate with:

    python3 test/data/make-mini-lora.py

This is only a test fixture; it is not a useful fine-tune.
"""
import os
import struct

ALIGN = 32
GGML_TYPE_F32 = 0   # tensor storage type (ggml_type enum)
GGUF_TYPE_F32 = 6   # metadata value type (gguf_type enum)
GGUF_TYPE_STRING = 8

BASE_NAME = "blk.0.attn_q.weight"
N_EMBD = 64
RANK = 1

KVS = [
    ("general.architecture", GGUF_TYPE_STRING, "llama"),
    ("general.type", GGUF_TYPE_STRING, "adapter"),
    ("general.name", GGUF_TYPE_STRING, "mini-lora"),
    ("adapter.type", GGUF_TYPE_STRING, "lora"),
    ("adapter.lora.alpha", GGUF_TYPE_F32, 1.0),
]

TENSORS = [
    (BASE_NAME + ".lora_a", [N_EMBD, RANK], [1.0] * (N_EMBD * RANK)),
    (BASE_NAME + ".lora_b", [RANK, N_EMBD], [1.0] * (RANK * N_EMBD)),
]


def gguf_string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv_bytes(key, kind, value):
    out = gguf_string(key) + struct.pack("<I", kind)
    if kind == GGUF_TYPE_STRING:
        out += gguf_string(value)
    elif kind == GGUF_TYPE_F32:
        out += struct.pack("<f", value)
    else:
        raise ValueError(kind)
    return out


def main():
    data = bytearray()
    infos = bytearray()
    for name, dims, values in TENSORS:
        data += b"\x00" * (-len(data) % ALIGN)
        infos += gguf_string(name)
        infos += struct.pack("<I", len(dims))
        infos += struct.pack("<" + "Q" * len(dims), *dims)
        infos += struct.pack("<I", GGML_TYPE_F32)
        infos += struct.pack("<Q", len(data))
        data += struct.pack("<" + "f" * len(values), *values)

    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", 3, len(TENSORS), len(KVS))
    for key, kind, value in KVS:
        out += kv_bytes(key, kind, value)
    out += infos
    out += b"\x00" * (-len(out) % ALIGN)
    out += data

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mini-lora.gguf")
    with open(path, "wb") as f:
        f.write(out)
    print("wrote %s (%d bytes)" % (path, len(out)))


if __name__ == "__main__":
    main()
