# DiffusionGemma INT4-AWQ tensor mapping

Checkpoint: `models/diffusiongemma-26B-A4B-it-AWQ-INT4`  
Format: compressed-tensors `pack-quantized`, symmetric INT4, group size 32.

| Model role | Checkpoint tensors | Checkpoint dtype/layout | Runtime representation | Fusion use |
|---|---|---|---|---|
| tied embedding / LM head | `model.decoder.embed_tokens.weight` | F16 `[262144,2816]` | BF16 table on GPU 0 | unchanged |
| self-condition gate/up/down | `model.decoder.self_conditioning.{gate,up,down}_proj.weight` | F16; explicitly ignored by AWQ config | BF16; gate/up concatenated | existing one-GEMM gate/up; new add+norm terminal fusion |
| layer Q/K/V/O | `model.decoder.layers.L.self_attn.P.weight_packed`, `.weight_scale` | I32 nibbles `[N,K/8]`, F16 scales `[N,K/32]` | rebased s4 row-major weights, BF16 scales `[K/32,N]` | existing QKV/QK concatenated INT4 projection |
| shared dense gate/up/down | `model.decoder.layers.L.mlp.{gate,up,down}_proj.weight` | F16; explicitly ignored by AWQ config | BF16 | new optional gate/up row concatenation `[4224,2816]` |
| router projection | `model.decoder.layers.L.router.proj.weight` | F16; explicitly ignored by AWQ config | BF16 | unchanged |
| router scales | `router.scale`, `router.per_expert_scale` | F16 | BF16 channel scale; FP32 expert gains | feeds existing fused triple prenorm / device top-k |
| expert gate/up | `model.decoder.layers.L.experts.E.{gate,up}_proj.{weight_packed,weight_scale}` | two s4 logical matrices `[704,2816]`, group 32 | one concatenated `Int4Linear` `[1408,2816]` per expert | existing one INT4 gate/up GEMM; terminal output feeds new postnorm fusion |
| expert down | `model.decoder.layers.L.experts.E.down_proj.{weight_packed,weight_scale}` | s4 logical `[2816,704]`, group 32 | one `Int4Linear` per expert | unchanged GEMM; output consumed directly by new terminal fusion |
| layer norms | `model.decoder.layers.L.*layernorm.weight` | F16 `[2816]` | BF16 | triple prenorm and expert terminal postnorm |
| layer scalar | decoder and encoder `layer_scalar` | F16/BF16 scalar | host float | folded into terminal postnorm |

The loader still validates each packed projection's logical K/N and derives
group size from the scale tensor rather than hardcoding 32.

