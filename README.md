# ggufdiff

[![tests](https://github.com/Kiwi172/ggufdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/Kiwi172/ggufdiff/actions/workflows/ci.yml)

**What actually changed between two GGUF files?**

You have `model-Q4_K_M.gguf` and `model-IQ3_XXS.gguf`. Same name, 8 GB
apart. Which layers got hit harder? Did the quantization touch
attention or only the FFN? Are the embeddings still f32? Is the KV
geometry identical — or is one of these secretly a hybrid model where
only every fourth layer carries a cache? And are these even the same
model, or did someone rename a different checkpoint?

`ggufdiff` reads both headers (never the weight data) and answers all
of it:

```
A: Qwen3.6 35B A3B  ·  qwen35moe  ·  40 layers  ·  21.16 GB  ·  4.88 bits/weight
B: Qwen3.6 35B A3B  ·  qwen35moe  ·  40 layers  ·  13.61 GB  ·  3.14 bits/weight

size: A  21.16 GB  ->  B  13.61 GB   (-7.54 GB, -35.7%)
kv/token (f16): A  20.5 kB  B  20.5 kB

structure identical: same arch, layers, heads, vocab -- a true quant-vs-quant comparison

bytes by format (A -> B):
  q4_k        16.03 GB ->  289.0 MB
  q6_k         5.04 GB ->    0.0 kB      <- attention was q6_k in A
  iq3_xxs        0.0 kB -> 12.57 GB      <- and iq3_xxs in B
  q5_k           0.0 kB ->  349.6 MB     <- attention partially kept at q5_k

per-layer sizes (40 of 40 layers changed):
  blk.0     549.8 MB ->  327.6 MB  -222.1 MB  format: q4_k -> iq3_xxs
  ...
```

That format table is the real story of a quant: llama.cpp's importance
matrix keeps attention weights at higher precision than the FFN bulk,
and ggufdiff shows you exactly how each publisher applied it.

One file, Python standard library only. It reads headers and file
metadata — a 20 GB model takes milliseconds.

---

## Install

Any machine with Python 3.9 or newer:

```bash
curl -sO https://raw.githubusercontent.com/Kiwi172/ggufdiff/main/ggufdiff.py
chmod +x ggufdiff.py
```

Or:

```bash
pipx install git+https://github.com/Kiwi172/ggufdiff
```

## Use

```bash
ggufdiff model-Q4_K_M.gguf model-IQ3_XXS.gguf
ggufdiff a.gguf b.gguf --all-layers        # every layer row
ggufdiff a.gguf b.gguf --json | jq .size_delta
```

If the two files are not the same model, ggufdiff says so and lists
what differs (architecture, layer count, embedding width, KV heads,
head size, vocab, attention interval) — because a "quant comparison"
between two different checkpoints is a silent lie.

## What it computes

| output | how |
|---|---|
| exact weight bytes, per tensor and per layer | data-section offsets (correct for every quant format, including MXFP4/NVFP4) |
| effective bits-per-weight | weight bytes × 8 ÷ total element count (from tensor dims) |
| bytes by quant format | dominant-type accounting per file |
| per-layer deltas + format changes | tensor names (`blk.N.*`) grouped |
| KV bytes per token | `2 × kv_layers × kv_heads × head_size × 2`, honouring `full_attention_interval` for hybrid models |
| tensor set differences | tensors present in only one file (untied embeddings, extra adapters) |

## Non-goals

- **Not a quality benchmark.** Bits-per-weight is a proxy, not a score.
  Run your own evals for that.
- **Not ggufcheck's replacement.** "Will it fit?" is ggufcheck's job;
  this tool tells you what the two files *are*. The fit verdict for
  either file is one `ggufcheck` call away.
- **Headers only.** Weight bytes are measured by offsets, never read.

## Tests

```bash
python3 -m pytest tests/      # or: python3 tests/test_ggufdiff.py
```

The tests build synthetic GGUF files byte by byte, so every size,
bits-per-weight, KV geometry, and delta is checked against arithmetic
the test itself controls — including the hybrid-attention interval and
the "not the same model" guard.

## License

MIT
