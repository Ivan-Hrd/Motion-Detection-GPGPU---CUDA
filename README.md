# GPU Accelerated Motion Detection in CUDA

Using : C++, CUDA, Nsight Systems, Nsight Compute, Git

## Overview

This project implements a motion detection filter, first developed as a CPU baseline to validate correctness, then rewritten as a full CUDA pipeline to run on GPU.

## Development Process

- Developed a baseline filter running on CPU first to ensure motion detection works correctly.
- Rewrote the full filter pipeline in CUDA to run on GPU.
- Performed overview analysis with Nsight Systems to identify the bottleneck in the pipeline.
- Performed kernel-level performance analysis with Nsight Compute to determine whether the bottleneck was memory-bound or compute-bound.

## Optimizations

- Optimized memory transfers between host and device.
- Converted data layout from array of structures (AoS) to structure of arrays (SoA) to enable coalesced memory access, resulting in a 1.37x speedup.
- Replaced the random number generation approach with a counter-based pseudorandom number generator to reduce register usage, resulting in a 1.14x speedup.

## Tools

- **C++** for the CPU baseline and host-side code
- **CUDA** for GPU kernel implementation
- **Nsight Systems** for pipeline-level performance analysis
- **Nsight Compute** for kernel-level performance analysis
- **Git** for version control
