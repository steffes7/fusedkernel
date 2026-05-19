# fusedkernel
CompScis 657s final project on a Triton Fused Kernel.

# Fusing LayerNorm: A Memory-Efficient Triton Kernel
**CS 790/657 — Domain-Specific Programming for AI | Austin Steffes**

## Dependencies
- Python 3.10+
- CUDA 12.x
- PyTorch
- Triton

Install:
```bash
pip install torch triton
```

## Reproduce Experiments
Run in Google Colab with a GPU runtime (Tesla T4 recommended):

```bash
pip install triton
python fused_layernorm.py
```

This will run correctness tests across hidden dims 768–8192, then the full benchmark sweep across 12 configurations.

## Expected Output
CORRECTNESS TESTS
hidden_dim=  768 | tutorial ✓ | fused ✓ | autotuned ✓
...
hidden_dim= 8192 | tutorial ✓ | fused ✓ | autotuned ✓
All correctness tests passed!
BENCHMARKS
Config | Naive ms | PyTorch ms | Compiled ms | Tutorial ms | Fused ms | vs Naive | vs PyTorch | Fused BW GB/s
...

## Directory Structure
.
├── fused_layernorm.py   # Main implementation and benchmarks
└── README.md

## What's Inside
- **Section 1** — Naive PyTorch baseline (3-pass)
- **Section 2** — Official Triton tutorial kernel (reference)
- **Section 3** — Fused Welford kernel (primary contribution)
- **Section 4** — Correctness tests
- **Section 5** — Benchmarking harness
- **Section 6** — Memory tracker
