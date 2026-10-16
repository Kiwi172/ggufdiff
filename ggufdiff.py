#!/usr/bin/env python3
"""ggufdiff -- what actually changed between two GGUF files?

You have model-Q4_K_M.gguf and model-IQ3_XXS.gguf. Same name, 8 GB
apart. Which layers got hit harder? Did the quantization touch
attention or only the FFN? Are the embeddings still f32? Is the KV
geometry identical? Are these even the same model, or did someone
rename a different checkpoint?

ggufdiff reads both headers (never the weights) and answers:
size and effective bits-per-weight, per-quant-format byte totals,
per-layer size deltas, tensor sets present in only one file, and
structural mismatches (hparams, vocab, layer count) that mean "these
are not two quants of the same model". Stdlib-only, no llama.cpp.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

VERSION = "0.1.0"

GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k",
    14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
    19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 39: "mxfp4",
    40: "nvfp4", 41: "q1_0", 42: "q2_0",
}

BLK_RE = re.compile(r"^blk\.(\d+)\.")


class GgufError(Exception):
    pass


def _rd_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 1 << 28:
        raise GgufError("implausible string length; not a GGUF?")
    return f.read(n).decode("utf-8", "replace")


def _rd_val(f, t):
    FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
    if t == 8:
        return _rd_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_rd_val(f, et) for _ in range(n)]
    if t not in FMT:
        raise GgufError(f"unknown GGUF value type {t}")
    fmt = FMT[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


class Model:
    """GGUF header: metadata + per-tensor sizes (from data offsets)."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise GgufError("not a GGUF file")
            (self.version,) = struct.unpack("<I", f.read(4))
            if self.version > 3:
                raise GgufError(f"GGUF version {self.version} unsupported")
            (self.n_tensors,) = struct.unpack("<Q", f.read(8))
            (n_kv,) = struct.unpack("<Q", f.read(8))
            self.meta = {}
            for _ in range(n_kv):
                k = _rd_str(f)
                (t,) = struct.unpack("<I", f.read(4))
                self.meta[k] = _rd_val(f, t)
            infos = []
            for _ in range(self.n_tensors):
                name = _rd_str(f)
                (nd,) = struct.unpack("<I", f.read(4))
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
                (ty,) = struct.unpack("<I", f.read(4))
                (off,) = struct.unpack("<Q", f.read(8))
                infos.append((name, dims, ty, off))
            hdr_end = f.tell()
        infos.sort(key=lambda x: x[3])
        fsize = path.stat().st_size
        align = int(self.meta.get("general.alignment", 32))
        data_start = (hdr_end + align - 1) // align * align

        self.tensors = {}          # name -> {bytes, elems, type}
        self.layer_bytes = {}
        self.type_bytes = {}
        self.elements = 0
        self.weights = 0
        for i, (name, dims, ty, off) in enumerate(infos):
            end = infos[i + 1][3] if i + 1 < len(infos) else fsize - data_start
            size = end - off
            elems = 1
            for d in dims:
                elems *= d
            tname = GGML_TYPE_NAMES.get(ty, f"type{ty}")
            self.tensors[name] = {"bytes": size, "elems": elems, "type": tname}
            self.weights += size
            self.elements += elems
            self.type_bytes[tname] = self.type_bytes.get(tname, 0) + size
            m = BLK_RE.match(name)
            if m:
                self.layer_bytes[int(m.group(1))] = \
                    self.layer_bytes.get(int(m.group(1)), 0) + size

        a = self.meta.get("general.architecture")
        if not a:
            raise GgufError("no general.architecture in metadata")
        self.arch = a
        g = lambda key, default=None: self.meta.get(f"{a}.{key}", default)
        self.name = self.meta.get("general.name", path.stem)
        self.n_layer = int(g("block_count") or 0)
        self.n_embd = int(g("embedding_length") or 0)
        self.head_kv = int(g("attention.head_count_kv")
                           or g("attention.head_count") or 0)
        self.head_size = int(g("attention.key_length") or 0)
        self.vocab = int(g("vocab_size") or 0)
        self.interval = int(g("full_attention_interval") or 1)

    @property
    def bpw(self):
        """Effective bits per weight (offset-based, exact for any format)."""
        if not self.elements:
            return 0.0
        return self.weights * 8 / self.elements

    def kv_per_token(self):
        kv_layers = (self.n_layer if self.interval <= 1 else
                     sum(1 for i in range(self.n_layer)
                         if (i + 1) % self.interval == 0))
        return 2 * kv_layers * self.head_kv * self.head_size * 2  # f16


# ---------------------------------------------------------------------------
# diffing
# ---------------------------------------------------------------------------

def structural_mismatches(a, b):
    out = []
    if a.arch != b.arch:
        out.append(f"different architectures: {a.arch} vs {b.arch}")
    for label, va, vb in [("layer count", a.n_layer, b.n_layer),
                          ("embedding width", a.n_embd, b.n_embd),
                          ("kv heads", a.head_kv, b.head_kv),
                          ("head size", a.head_size, b.head_size),
                          ("vocab size", a.vocab, b.vocab),
                          ("attention interval", a.interval, b.interval)]:
        if va != vb and va and vb:
            out.append(f"{label} differs: {va} vs {vb}")
    return out


def tensor_set_diff(a, b):
    only_a = sorted(set(a.tensors) - set(b.tensors))
    only_b = sorted(set(b.tensors) - set(a.tensors))
    return only_a, only_b


def per_layer_delta(a, b):
    rows = []
    for layer in sorted(set(a.layer_bytes) | set(b.layer_bytes)):
        ba, bb = a.layer_bytes.get(layer, 0), b.layer_bytes.get(layer, 0)
        rows.append((layer, ba, bb, bb - ba))
    return rows


def dominant_type_per_layer(a, b):
    """For each layer, the format holding most bytes in each file."""
    def per_layer_types(m):
        d = {}
        for name, t in m.tensors.items():
            mm = BLK_RE.match(name)
            if mm:
                layer = int(mm.group(1))
                cell = d.setdefault(layer, {})
                cell[t["type"]] = cell.get(t["type"], 0) + t["bytes"]
        return {k: max(v, key=v.get) for k, v in d.items()}
    return per_layer_types(a), per_layer_types(b)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def fmt(n):
    if n >= 1e9:
        return f"{n / 1e9:6.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:6.1f} MB"
    return f"{n / 1e3:6.1f} kB"


def build_parser():
    ap = argparse.ArgumentParser(
        prog="ggufdiff",
        description="What actually changed between two GGUF files? Sizes, "
                    "effective bits-per-weight, per-format byte totals, "
                    "per-layer deltas, tensor set differences, structural "
                    "sanity checks. Reads headers only. Stdlib-only.")
    ap.add_argument("a", metavar="GGUF_A")
    ap.add_argument("b", metavar="GGUF_B")
    ap.add_argument("--layers", type=int, default=12,
                    help="show at most N per-layer rows (default 12)")
    ap.add_argument("--all-layers", action="store_true",
                    help="show every layer row")
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--version", action="version", version=f"ggufdiff {VERSION}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        a, b = Model(Path(args.a)), Model(Path(args.b))
    except (GgufError, OSError) as e:
        print(f"ggufdiff: {e}", file=sys.stderr)
        return 2

    mismatches = structural_mismatches(a, b)
    only_a, only_b = tensor_set_diff(a, b)
    ta, tb = dominant_type_per_layer(a, b)

    if args.as_json:
        print(json.dumps({
            "tool": "ggufdiff", "version": VERSION,
            "same_model": not mismatches,
            "mismatches": mismatches,
            "a": {"path": str(a.path), "name": a.name, "arch": a.arch,
                  "weights": a.weights, "bpw": round(a.bpw, 3),
                  "types": a.type_bytes, "kv_per_token": a.kv_per_token()},
            "b": {"path": str(b.path), "name": b.name, "arch": b.arch,
                  "weights": b.weights, "bpw": round(b.bpw, 3),
                  "types": b.type_bytes, "kv_per_token": b.kv_per_token()},
            "size_delta": b.weights - a.weights,
            "only_in_a": only_a[:50], "only_in_b": only_b[:50],
            "layers": [{"layer": l, "a": ba, "b": bb, "delta": d}
                       for l, ba, bb, d in per_layer_delta(a, b)],
        }, indent=1))
        return 0

    def header(m):
        return (f"{m.name}  ·  {m.arch}  ·  {m.n_layer} layers  ·  "
                f"{fmt(m.weights)}  ·  {m.bpw:.2f} bits/weight")

    print("A: " + header(a))
    print("B: " + header(b))
    delta = b.weights - a.weights
    print(f"\nsize: A {fmt(a.weights)}  ->  B {fmt(b.weights)}   "
          f"({'+' if delta > 0 else ''}{fmt(abs(delta))}, "
          f"{100 * delta / max(a.weights, 1):+.1f}%)")
    print(f"kv/token (f16): A {fmt(a.kv_per_token())}  "
          f"B {fmt(b.kv_per_token())}")

    if mismatches:
        print("\n" + "NOT the same model — these files differ structurally:")
        for mm in mismatches:
            print(f"  ! {mm}")
    else:
        print("\nstructure identical: same arch, layers, heads, vocab "
              "-- a true quant-vs-quant comparison")

    print("\nbytes by format (A -> B):")
    for t in sorted(set(a.type_bytes) | set(b.type_bytes)):
        va, vb = a.type_bytes.get(t, 0), b.type_bytes.get(t, 0)
        print(f"  {t:<10} {fmt(va)} -> {fmt(vb)}   "
              f"({'+' if vb >= va else ''}{fmt(abs(vb - va))})")

    rows = per_layer_delta(a, b)
    changed = [r for r in rows if r[1] != r[2]]
    show = rows if args.all_layers else rows[:args.layers]
    print(f"\nper-layer sizes ({len(changed)} of {len(rows)} layers changed):")
    for layer, ba, bb, d in show:
        mark = ""
        if ta.get(layer) and tb.get(layer) and ta[layer] != tb[layer]:
            mark = f"  format: {ta[layer]} -> {tb[layer]}"
        print(f"  blk.{layer:<4} {fmt(ba)} -> {fmt(bb)}  "
              f"{'+' if d >= 0 else '-'}{fmt(abs(d))}{mark}")
    if len(rows) > len(show):
        print(f"  … {len(rows) - len(show)} more (use --all-layers)")

    if only_a or only_b:
        print("\ntensor set differences:")
        for t in only_a[:10]:
            print(f"  only in A: {t}")
        for t in only_b[:10]:
            print(f"  only in B: {t}")
    print("\nheaders only: ggufdiff never reads the weight data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
