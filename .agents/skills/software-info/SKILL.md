---
name: software-info
description: "SYCL software optimization techniques for Intel Xe2-HPG GPUs. USE FOR: memory access patterns (vec_load, coalesced access, prefetch), sub-group operations, SLM tiling, occupancy tuning, kernel launch configuration. Includes benchmarks to measure real throughput for each technique."
---

# SYCL Software Optimization Guide (Xe2-HPG)

This skill documents key software optimization techniques for Intel GPU SYCL kernels, with runnable benchmarks to measure actual performance impact.

## Prerequisites

- Intel oneAPI DPC++ compiler (`icpx`) with `-fsycl`
- Intel GPU with Level-Zero runtime
- Hardware info from `hardware_info/device_info` (for roofline context)

## 1. Memory Access Patterns

### 1.1 Vector Load (`vec_load`)

**Why**: A single `sycl::vec<T,N>` load issues one LSC (Load/Store Cache) message for N elements instead of N separate scalar loads. This reduces message overhead and improves cache line utilization.

**Rules of thumb**:
- Choose a vector width that matches the natural transaction or register granularity of the target device
- Effective bandwidth approaches peak when load width matches cache line granularity
- Best for streaming patterns where each element is accessed once

**Compile & Run**:
```bash
cd <repo-root>/software_info
icpx -fsycl -O2 -o mem_access_bench mem_access_bench.cpp
./mem_access_bench
```

### 1.2 Coalesced Access

**Why**: When work-items in a sub-group access consecutive addresses, the hardware merges them into fewer memory transactions.

| Pattern | Transactions |
|---------|--------------|
| Coalesced | Fewest transactions for the target access width |
| Small stride | More transactions than coalesced access |
| Large stride | Often degenerates toward one transaction per lane |
| Random | Typically the least efficient |

### 1.3 Prefetch

**Why**: Hide memory latency by issuing loads ahead of time. Useful when access pattern is predictable but data is not in cache.

Use a device-supported prefetch mechanism when access is predictable and the data is not already resident in cache.

### 1.4 Aligned Access

**Why**: Misaligned loads may cross cache lines, causing extra transactions.

```cpp
float* aligned_ptr = sycl::aligned_alloc_device<float>(64, N, q); // 64B aligned
```

## 2. Sub-group (SIMD) Operations

### 2.1 Shuffle / Broadcast / Reduce

Sub-group operations happen at register level - no memory traffic:

```cpp
auto sg = it.get_sub_group();
float val = sg.shuffle(x, 0);           // broadcast lane 0
float sum = sycl::reduce_over_group(sg, x, sycl::plus<>()); // reduction
float neighbor = sg.shuffle_xor(x, 1);  // swap with adjacent lane
```

### 2.2 SIMD Width Selection

| Width | Pros | Cons |
|-------|------|------|
| 16 | More sub-groups/WG, better occupancy | Less work per instruction |
| 32 | Wider vector ops, fewer instructions | Fewer sub-groups, may reduce occupancy |

Control with: `[[intel::reqd_sub_group_size(16)]]`

## 3. SLM (Shared Local Memory) Tiling

### 3.1 When to Use

- Data reuse across work-items in the same work-group (e.g., matrix multiply tiles)
- Reduction / scan within a work-group
- Avoid when data is only used once (SLM adds write+read overhead)

### 3.2 Bank Conflict Avoidance

SLM has 32 banks (4B each). Conflicts cause serialization.

```cpp
// BAD: stride-32 access -> all hit same bank
slm[lid * 32]

// GOOD: add padding to break conflict pattern
constexpr int PAD = 1;
slm[lid * (32 + PAD)]
```

### 3.3 Capacity vs Occupancy Trade-off

| SLM per WG | WGs per Xe-core (128KB total) | Occupancy |
|------------|-------------------------------|-----------|
| 16 KB | 8 | High |
| 32 KB | 4 | Medium |
| 64 KB | 2 | Low |
| 128 KB | 1 | Minimum |

## 4. Occupancy & Register Pressure

### 4.1 GRF Modes

| Mode | Registers/thread | Threads/XVE | Best for |
|------|-----------------|-------------|----------|
| Default (128 GRF) | 128 x 32B = 4KB | 8 | Memory-bound (high occupancy hides latency) |
| Large (256 GRF) | 256 x 32B = 8KB | 4 | Compute-bound (more registers, less spilling) |

Control with: `-ftarget-register-alloc-mode=device:auto` or `[[intel::reqd_work_group_size(...)]]`

### 4.2 Decision Logic

```
if kernel is memory-bound (AI < Ridge Point):
    -> prefer high occupancy (default GRF, more threads)
    -> maximize memory-level parallelism to hide latency
else:
    -> prefer large GRF (avoid register spills to memory)
    -> reduce thread count is acceptable
```

## 5. Kernel Launch Configuration

### 5.1 Work-group Size

- Must be multiple of sub-group size (16 or 32)
- Recommended: 256 or 512 for most kernels
- Smaller WG -> more WGs -> better load balancing across Xe-cores
- Larger WG -> more SLM available per work-item -> better for tiling

### 5.2 Total Work-items

- Should be >> XVE count x threads/XVE (160 x 8 = 1280 minimum to saturate)
- Ideally 10-100x more for latency hiding

## 6. Compiler Hints

```cpp
#pragma unroll 4                              // loop unrolling
[[intel::loop_coalesce(2)]]                   // merge nested loops
[[intel::reqd_sub_group_size(16)]]            // fix SIMD width
[[intel::kernel_args_restrict]]               // no-alias hint
```

## 7. Sub-group Lane Packing & reqd_sub_group_size

### 7.1 Lane Packing for Multi-token Processing

When packing multiple data items (tokens, matrix rows, tiles) into a single sub-group, each item should occupy a contiguous block of lanes:

```cpp
constexpr int ITEM_LANES = /* lanes needed for one logical item */;
constexpr int TOKS_PER_ROUND = SG / ITEM_LANES;   // items per round
int my_item_in_round = lane / ITEM_LANES;
int my_idx = lane % ITEM_LANES;
```

**Critical rule**: `permute_group_by_xor(sg, val, offset)` operates across all lanes in the sub-group. When multiple items share a sub-group, XOR offsets must stay within the lane block for one item.

| SG Size | Item lanes | Items/Round | Safe XOR range |
|---------|------------|-------------|----------------|
| SG | ITEM_LANES | SG / ITEM_LANES | off < ITEM_LANES |

### 7.2 reqd_sub_group_size Correctness Checklist

Before adding `[[sycl::reqd_sub_group_size(N)]]`, verify:

1. **Lane-split logic** — If code uses lane ranges to split work between items, ensure the split boundaries are compatible with the chosen sub-group size.
2. **reduce_over_group** — Works for any SG size (SYCL runtime handles it).
3. **permute_group_by_xor** — XOR offsets must stay within logical item boundaries.
4. **Work-group reduction scratch** — If `COUNT > SG_SIZE`, the cross-SG reduction must loop: `for (j = sg_lid; j < COUNT; j += SG_SIZE)` instead of `if (sg_lid < COUNT)`.

### 7.3 Work-group Reduction with COUNT > SG_SIZE

```cpp
// BAD: silently drops values when COUNT > SG_SIZE
if (sg_id == 0 && sg_lid < COUNT) {
    scratch[sg_lid] = sum_over_subgroups(...);
}

// GOOD: handles any COUNT
if (sg_id == 0) {
    for (int j = sg_lid; j < COUNT; j += SG_SIZE) {
        float t = 0.f;
        for (int s = 0; s < N_SG; ++s)
            t += scratch[s * COUNT + j];
        scratch[j] = t;
    }
}
```

In general, use a strided loop whenever the reduction count can exceed the sub-group width.

## Key Takeaways

1. **vec_load** with a width matched to the device granularity can approach peak bandwidth
2. **Coalesced access** is critical - stride > 1 causes severe bandwidth loss
3. **Sub-group ops** are free (register-level) - use them for reductions and data sharing
4. **SLM** trades occupancy for data reuse - only worth it when reuse factor > 2x
5. **Memory-bound kernels** (most DL inference ops) benefit more from occupancy than from large GRF
6. **reqd_sub_group_size** improves consistency and prevents compiler from auto-selecting wider SIMD, but requires verifying all lane-packing logic matches the chosen width
7. **intel::kernel_args_restrict** is a low-risk hint when aliasing is already absent
