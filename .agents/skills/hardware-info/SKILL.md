---
name: hardware-info
description: "Measure Intel GPU hardware specifications using SYCL. USE FOR: querying GPU device info, memory hierarchy, bandwidth estimates, compute throughput, roofline analysis ridge points, architecture constants for Xe2-HPG. Compiles and runs device_info.cpp to produce a full hardware report."
---

# Hardware Info Measurement

This skill compiles and runs a SYCL program to measure Intel GPU hardware specifications, including compute units, memory hierarchy, theoretical bandwidth, peak compute, and roofline ridge points.

## Prerequisites

- Intel oneAPI DPC++ compiler (`icpx`) available in PATH
- Intel GPU with Level-Zero runtime installed
- Environment sourced: `source /opt/intel/oneapi/setvars.sh` (or equivalent)

## Steps

### 1. Compile the device_info program

```bash
cd /mnt/disk3/home/xiaolong/sycl-skills/hardware_info
icpx -fsycl -o device_info device_info.cpp
```

### 2. Run the program

```bash
./device_info
```

### 3. Interpret the output

The program reports the following sections:

| Section | Key Metrics |
|---------|-------------|
| **Device Info** | GPU name, vendor, driver version, XVE count, Xe-cores, max clock, sub-group sizes |
| **Memory Hierarchy** | Global memory (VRAM), L2 cache size, SLM per Xe-core, cache line size |
| **Bandwidth (Theoretical Peak)** | VRAM BW, L2 BW, SLM BW, GRF BW |
| **Compute (Theoretical Peak)** | FP32 TFLOPS, FP16 TFLOPS, INT8 TOPS |
| **Roofline Key Points** | Ridge point for VRAM / L2 / SLM (FLOP/Byte threshold for compute-bound vs memory-bound) |
| **Architecture Constants** | GRF per thread, register width, threads per XVE, L1 cache estimate |

### 4. Using the results for kernel optimization

- **Arithmetic Intensity > Ridge Point** → kernel is compute-bound → optimize ALU utilization
- **Arithmetic Intensity < Ridge Point** → kernel is memory-bound → optimize data locality / reduce memory traffic
- Use the bandwidth hierarchy (GRF >> SLM >> L2 >> VRAM) to decide which memory level to target for data reuse

## Output Example

The following values are illustrative for an Intel Xe2-HPG-class device; treat them as example output rather than a fixed target.

```
============================================
  Device Info (Card 0, all cards identical)
============================================
Name:                 Intel(R) Arc(TM) ...
Max Compute Units:    160 (= XVE count)
Xe-cores:             20 (= XVE / 8)
Max Clock Freq(MHz):  2850

============================================
  Bandwidth (Theoretical Peak)
============================================
VRAM Bandwidth:       456.0 GB/s (per card)
L2 BW (total):        3648.0 GB/s
SLM Total BW (all):   3648.0 GB/s (read)
GRF BW (total):       58.4 TB/s

============================================
  Compute (Theoretical Peak)
============================================
FP32:                 7.3 TFLOPS
FP16:                 14.6 TFLOPS

============================================
  Roofline Key Points (FP32)
============================================
Ridge Point (VRAM):   16.0 FLOP/Byte
Ridge Point (L2):     2.0 FLOP/Byte
Ridge Point (SLM):    2.0 FLOP/Byte
```

## Notes

- The program auto-detects hardware via Level-Zero backend and filters out duplicate OpenCL devices.
- If the driver returns invalid VRAM bus width or clock rate, it falls back to known platform defaults when available.
- All bandwidth/compute numbers are **theoretical peaks**; real-world performance will be lower due to occupancy, bank conflicts, etc.
