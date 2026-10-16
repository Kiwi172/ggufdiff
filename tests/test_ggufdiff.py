"""Tests for ggufdiff. Run with: python3 -m pytest tests/ (or python3 tests/test_ggufdiff.py)

Synthetic GGUF files built byte-by-byte, so every delta is checked
against arithmetic the test controls.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

TOOL = str(Path(__file__).resolve().parent.parent / "ggufdiff.py")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ggufdiff as gd  # noqa: E402


def _w_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _w_kv(k, t, v):
    out = _w_str(k) + struct.pack("<I", t)
    if t == 4:
        out += struct.pack("<I", v)
    elif t == 8:
        out += _w_str(v)
    return out


def build(path: Path, arch="testarch", n_layer=3, layer_bytes=None,
          outer=1024, vocab=1000, embd=256, head_kv=2, head_size=64,
          interval=1, name="Test Model"):
    """f16 tensors; layer_bytes = bytes per blk tensor per layer."""
    if layer_bytes is None:
        layer_bytes = [2048] * n_layer
    metas = [
        _w_kv("general.architecture", 8, arch),
        _w_kv("general.name", 8, name),
        _w_kv(f"{arch}.block_count", 4, n_layer),
        _w_kv(f"{arch}.embedding_length", 4, embd),
        _w_kv(f"{arch}.attention.head_count", 4, 4),
        _w_kv(f"{arch}.attention.head_count_kv", 4, head_kv),
        _w_kv(f"{arch}.attention.key_length", 4, head_size),
        _w_kv(f"{arch}.vocab_size", 4, vocab),
        _w_kv(f"{arch}.context_length", 4, 4096),
    ]
    if interval > 1:
        metas.append(_w_kv(f"{arch}.full_attention_interval", 4, interval))
    tensors = []
    for i in range(n_layer):
        tensors.append((f"blk.{i}.ffn.weight", layer_bytes[i] // 2))
        if i == 1:
            tensors.append((f"blk.{i}.attn.weight", 512))  # extra tensor
    tensors.append(("token_embd.weight", outer // 2))
    hdr = b"GGUF" + struct.pack("<I", 3)
    hdr += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metas))
    hdr += b"".join(metas)
    infos, data, cur = b"", b"", 0
    for tname, ne in tensors:
        infos += _w_str(tname) + struct.pack("<I", 1)
        infos += struct.pack("<Q", ne) + struct.pack("<I", 1)  # f16
        infos += struct.pack("<Q", cur)
        data += b"\x00" * (ne * 2)
        cur += ne * 2
    body = hdr + infos
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body + b"\x00" * ((-len(body)) % 32) + data)
    return path


def run(*argv):
    return subprocess.run([sys.executable, TOOL, *argv],
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------

def test_model_parses_types_and_bpw(tmp_path):
    p = build(tmp_path / "a.gguf", n_layer=2, layer_bytes=[2048, 2048],
              outer=1024)
    m = gd.Model(p)
    # builder: (name, n_elements); attn = 512 elems * 2 bytes = 1024
    assert m.weights == 2048 * 2 + 1024 + 1024
    assert m.type_bytes == {"f16": m.weights}
    assert m.bpw == 16.0                                # all f16
    assert m.layer_bytes[1] == 2048 + 1024


def test_kv_per_token_plain_and_hybrid(tmp_path):
    m = gd.Model(build(tmp_path / "a.gguf", n_layer=4))
    assert m.kv_per_token() == 2 * 4 * 2 * 64 * 2
    h = gd.Model(build(tmp_path / "h.gguf", n_layer=8, interval=4))
    assert h.kv_per_token() == 2 * 2 * 2 * 64 * 2      # only 2 kv layers


def test_structural_match_and_mismatch(tmp_path):
    a = gd.Model(build(tmp_path / "a.gguf"))
    b = gd.Model(build(tmp_path / "b.gguf"))
    assert gd.structural_mismatches(a, b) == []
    c = gd.Model(build(tmp_path / "c.gguf", n_layer=5,
                       layer_bytes=[1024] * 5))
    ms = gd.structural_mismatches(a, c)
    assert any("layer count" in x for x in ms)
    d = gd.Model(build(tmp_path / "d.gguf", arch="otherarch"))
    assert any("architectures" in x for x in gd.structural_mismatches(a, d))


def test_size_and_format_deltas(tmp_path):
    a = build(tmp_path / "a.gguf", n_layer=3, layer_bytes=[4096] * 3)
    b = build(tmp_path / "b.gguf", n_layer=3, layer_bytes=[2048] * 3)
    r = run(str(a), str(b))
    assert r.returncode == 0, r.stderr
    assert "structure identical" in r.stdout
    assert "3 of 3 layers changed" in r.stdout
    assert "f16" in r.stdout


def test_tensor_set_diff(tmp_path):
    a = build(tmp_path / "a.gguf")
    b = build(tmp_path / "b.gguf")
    # b has an extra attn tensor on blk.1; a (same builder) too -> equal.
    only_a, only_b = gd.tensor_set_diff(gd.Model(a), gd.Model(b))
    assert only_a == [] and only_b == []


def test_cli_json(tmp_path):
    a = build(tmp_path / "a.gguf", n_layer=2, layer_bytes=[4096, 4096])
    b = build(tmp_path / "b.gguf", n_layer=2, layer_bytes=[2048, 2048])
    r = run("--json", str(a), str(b))
    d = json.loads(r.stdout)
    assert d["same_model"] is True
    assert d["size_delta"] < 0
    assert d["a"]["bpw"] == 16.0
    assert len(d["layers"]) == 2
    assert d["layers"][0]["delta"] < 0


def test_cli_rejects_non_gguf(tmp_path):
    bad = tmp_path / "x.gguf"
    bad.write_bytes(b"nope" * 20)
    r = run(str(bad), str(bad))
    assert r.returncode == 2 and "not a GGUF" in r.stderr


if __name__ == "__main__":
    import inspect
    import tempfile
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                print(f"  ok  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
