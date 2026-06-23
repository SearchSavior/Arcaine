# Arcaine

Arcaine is a from scratch inference engine builting using SYCL + oneDNN meant to deliver bleeding edge performance on intel devices with important features missing everywhere else in the stack.

Note: this project makes heavy use of AI tools and generated code and may not be *artisinal*.


Initial release implements

- multi gpu expert parallel,
- multi gpu layer splitting and
- NVFP4 support for DiffusionGemma using hand-optimized SPIR-V FP4/FP8 rescaling kernels + latest oneDNN support NVFP4 matmul and reorder
- small chat cli
- produce html visualizations that replay denoising steps with diffusion-gemma

It's early days and the project is expected to move quickly- there are a ton of details to iron out.

## Performance

Arcaine currently implements DiffusionGemma


## Supported Models

- DiffusionGemma [BF16](https://huggingface.co/google/diffusiongemma-26B-A4B-it)/[NVFP4](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4)/[W4A16](https://huggingface.co/pixelkaiser/diffusiongemma-26B-A4B-it-AWQ-MLP-W4A16-G64-S32-L1024)

- Gemma4-12B [BF16](https://huggingface.co/google/gemma-4-12B-it)




## Container setup

```bash
export RENDER_GID=$(getent group render | cut -d: -f3)
docker compose build
docker compose run --rm \
  -v /mnt/Ironwolf-4TB/Projects/arcana/models:/workspace/models \
  dev   # interactive shell in /workspace
```

### Build

```bash
# Inside the dev container (see .devops/ for setup)
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j"$(nproc)"
```

The CMake targets add the NVFP4/DPAS SPIR-V translator extension at link time.
Do not pass `-Xspirv-translator` as a global compile flag; DPC++ will warn that
it is unused during normal host compilation.

For an AOT Battlemage build, add the SYCL target explicitly:

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx \
  -DARCAINE_SYCL_TARGETS=intel_gpu_bmg_g31
cmake --build build -j"$(nproc)"
```

Doing the build makes a few binaries which all accept `--help`.


Host requirements: Linux, Intel GPU, `i915`/`xe` driver, `/dev/dri` present,
user in `render` group.


## Notes

- oneDNN is built from source with `-DDNNL_CPU_RUNTIME=SYCL
  -DDNNL_GPU_RUNTIME=SYCL`. Mixing the binary distribution causes symbol
  conflicts — source build is required.
- If `diffusion_bench` reports an undefined oneDNN symbol such as
  `dnnl_primitive_attr_set_scales_v3`, re-run the CMake configure/build step so
  the binary embeds the `/opt/onednn/lib` runtime path ahead of oneAPI/OpenVINO
  library paths.
