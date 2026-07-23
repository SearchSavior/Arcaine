# Checkpoint tensor mapping

| Checkpoint family | Runtime owner | Format |
|---|---|---|
| `model.language_model.embed_tokens` | token embedding | BF16 |
| `model.language_model.layers.*.self_attn.*` | full-attention operator | E4M3 + BF16 scales/norms |
| `model.language_model.layers.*.linear_attn.*` | DeltaNet operator and cache | E4M3 + BF16/F32 parameters |
| `model.language_model.layers.0..55.mlp.*` | Xe2 NVFP4 MLP | packed U8 + FP8 block scales |
| `model.language_model.layers.56..63.mlp.*` | FP8 MLP | E4M3 + BF16 scales |
| `model.language_model.norm` | final RMS norm | BF16 |
| `lm_head` | vocabulary projection | E4M3 + BF16 scale |
| `model.visual.*` | Qwen vision tower | BF16 / E4M3 |
| `model.mtp.*` | resident MTP weights | BF16 |

`ShardedSafetensors` resolves tensors across all five shards. The loader tracks
every resolved name and reports missing, mismatched, or unconsumed tensors as a
hard error. Detailed names, shapes, dtypes, and shard locations are listed in
`tensors.tsv`.
