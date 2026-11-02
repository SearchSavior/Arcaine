---
name: sycl-optimization
description: "Intel GPU SYCL kernel optimization toolkit. USE FOR: writing high-performance SYCL kernels on Intel Xe2-HPG GPUs, performance analysis, roofline modeling, memory access optimization, sub-group operations, SLM tiling, occupancy tuning. Delegates to hardware-info (device specs & roofline) and software-info (optimization techniques & benchmarks)."
---

# SYCL Optimization Skills

A collection of skills for writing and optimizing SYCL kernels on Intel Xe2-HPG GPUs. This is the top-level entry point that routes to specialized sub-skills.

## Sub-skills

| Skill | Path | Use When |
|-------|------|----------|
| **hardware-info** | `hardware_info/SKILL.md` | Need device specs, memory hierarchy, bandwidth/compute peaks, roofline ridge points |
| **software-info** | `software_info/SKILL.md` | Need optimization techniques: vec_load, coalescing, prefetch, sub-group ops, SLM tiling, occupancy tuning |
| **register-info** | `register_info/SKILL.md` | Need to diagnose register spill, compare GRF modes, dump ISA, estimate register budget |

## Workflow

### Step 1: Gather hardware context (hardware-info)

Before optimizing a kernel, first run `hardware-info` to obtain:
- Peak bandwidth at each memory level (VRAM / L2 / SLM / GRF)
- Peak compute (FP32, FP16)
- Roofline ridge points

This tells you whether your kernel is **compute-bound** or **memory-bound**.

### Step 2: Apply optimization techniques (software-info)

Based on the roofline analysis:

| Kernel is… | Focus on |
|------------|----------|
| Memory-bound (AI < ridge point) | vec_load, coalesced access, prefetch, SLM tiling for reuse |
| Compute-bound (AI > ridge point) | Sub-group ops, SIMD width selection, occupancy tuning |

### Step 3: Benchmark and iterate

Use the benchmarks in `software_info/mem_access_bench.cpp` to measure real throughput and compare against theoretical peaks from Step 1.

## Quick Reference

```bash
# Compile & run hardware info
cd <repo-root>/hardware_info
icpx -fsycl -o device_info device_info.cpp && ./device_info

# Compile & run memory access benchmarks
cd <repo-root>/software_info
icpx -fsycl -O2 -o mem_access_bench mem_access_bench.cpp && ./mem_access_bench
```

## When to Use This Skill

- "How do I optimize this SYCL kernel?" → Start here
- "What are the hardware specs of this GPU?" → Route to `hardware-info`
- "How do I use vec_load / SLM / sub-group shuffle?" → Route to `software-info`
- "Is my kernel compute-bound or memory-bound?" → Run `hardware-info`, compute arithmetic intensity, compare to ridge point
