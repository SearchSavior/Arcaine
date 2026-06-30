# /mnt/Ironwolf-4TB/Projects/Arcaine/models/diffusiongemma-26B-A4B-it-Q8_0.gguf - GGUF Internal File Dump

- Endian: LITTLE endian

## Key Value Metadata Store

There are 47 key-value pairs in this file

| POS | TYPE      |  Count | Key                                              | Value                                                                                                                                                                                                      |
|----:|:----------|-------:|:-------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   1 | UINT32    |      1 | GGUF.version                                     | 3                                                                                                                                                                                                          |
|   2 | UINT64    |      1 | GGUF.tensor_count                                | 692                                                                                                                                                                                                        |
|   3 | UINT64    |      1 | GGUF.kv_count                                    | 44                                                                                                                                                                                                         |
|   4 | STRING    |      1 | general.architecture                             | `diffusion-gemma`                                                                                                                                                                                          |
|   5 | STRING    |      1 | general.type                                     | `model`                                                                                                                                                                                                    |
|   6 | STRING    |      1 | general.name                                     | `Dg_Rc0P1_Patched`                                                                                                                                                                                         |
|   7 | STRING    |      1 | general.size_label                               | `128x2.6B`                                                                                                                                                                                                 |
|   8 | UINT32    |      1 | diffusion-gemma.block_count                      | 30                                                                                                                                                                                                         |
|   9 | UINT32    |      1 | diffusion-gemma.context_length                   | 262144                                                                                                                                                                                                     |
|  10 | UINT32    |      1 | diffusion-gemma.embedding_length                 | 2816                                                                                                                                                                                                       |
|  11 | UINT32    |      1 | diffusion-gemma.feed_forward_length              | 2112                                                                                                                                                                                                       |
|  12 | UINT32    |      1 | diffusion-gemma.attention.head_count             | 16                                                                                                                                                                                                         |
|  13 | [INT32]   |     30 | diffusion-gemma.attention.head_count_kv          | [ 8, 8, 8, 8, 8, 2, 8, ... ]                                                                                                                                                                               |
|  14 | FLOAT32   |      1 | diffusion-gemma.rope.freq_base                   | 1e+06                                                                                                                                                                                                      |
|  15 | FLOAT32   |      1 | diffusion-gemma.rope.freq_base_swa               | 10000.0                                                                                                                                                                                                    |
|  16 | FLOAT32   |      1 | diffusion-gemma.attention.layer_norm_rms_epsilon | 1e-06                                                                                                                                                                                                      |
|  17 | UINT32    |      1 | diffusion-gemma.expert_count                     | 128                                                                                                                                                                                                        |
|  18 | UINT32    |      1 | diffusion-gemma.expert_used_count                | 8                                                                                                                                                                                                          |
|  19 | UINT32    |      1 | diffusion-gemma.attention.key_length             | 512                                                                                                                                                                                                        |
|  20 | UINT32    |      1 | diffusion-gemma.attention.value_length           | 512                                                                                                                                                                                                        |
|  21 | FLOAT32   |      1 | diffusion-gemma.final_logit_softcapping          | 30.0                                                                                                                                                                                                       |
|  22 | UINT32    |      1 | diffusion-gemma.attention.sliding_window         | 1024                                                                                                                                                                                                       |
|  23 | UINT32    |      1 | diffusion-gemma.attention.shared_kv_layers       | 0                                                                                                                                                                                                          |
|  24 | UINT32    |      1 | diffusion-gemma.embedding_length_per_layer_input | 0                                                                                                                                                                                                          |
|  25 | [BOOL]    |     30 | diffusion-gemma.attention.sliding_window_pattern | [ True, True, True, True, True, False, True, ... ]                                                                                                                                                         |
|  26 | UINT32    |      1 | diffusion-gemma.attention.key_length_swa         | 256                                                                                                                                                                                                        |
|  27 | UINT32    |      1 | diffusion-gemma.attention.value_length_swa       | 256                                                                                                                                                                                                        |
|  28 | UINT32    |      1 | diffusion-gemma.expert_feed_forward_length       | 704                                                                                                                                                                                                        |
|  29 | UINT32    |      1 | diffusion-gemma.rope.dimension_count             | 512                                                                                                                                                                                                        |
|  30 | UINT32    |      1 | diffusion-gemma.rope.dimension_count_swa         | 256                                                                                                                                                                                                        |
|  31 | BOOL      |      1 | diffusion-gemma.attention.causal                 | False                                                                                                                                                                                                      |
|  32 | UINT32    |      1 | diffusion.canvas_length                          | 256                                                                                                                                                                                                        |
|  33 | STRING    |      1 | tokenizer.ggml.model                             | `gemma4`                                                                                                                                                                                                   |
|  34 | [STRING]  | 262144 | tokenizer.ggml.tokens                            | [ `<pad>`, `<eos>`, `<bos>`, `<unk>`, `<mask>`, ... ]                                                                                                                                                      |
|  35 | [FLOAT32] | 262144 | tokenizer.ggml.scores                            | [ -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, ... ]                                                                                                                                     |
|  36 | [INT32]   | 262144 | tokenizer.ggml.token_type                        | [ 3, 3, 3, 3, 3, 1, 1, ... ]                                                                                                                                                                               |
|  37 | [STRING]  | 514906 | tokenizer.ggml.merges                            | [ `














`...`












 
`, `▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁`...`▁▁▁▁▁▁▁▁▁▁▁▁▁ ▁`, `															`...`													 	`, `














`...`











 

`, `▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁`...`▁▁▁▁▁▁▁▁▁▁▁▁ ▁▁`, ... ] |
|  38 | UINT32    |      1 | tokenizer.ggml.bos_token_id                      | 2                                                                                                                                                                                                          |
|  39 | UINT32    |      1 | tokenizer.ggml.eos_token_id                      | 1                                                                                                                                                                                                          |
|  40 | UINT32    |      1 | tokenizer.ggml.unknown_token_id                  | 3                                                                                                                                                                                                          |
|  41 | UINT32    |      1 | tokenizer.ggml.padding_token_id                  | 0                                                                                                                                                                                                          |
|  42 | UINT32    |      1 | tokenizer.ggml.mask_token_id                     | 4                                                                                                                                                                                                          |
|  43 | STRING    |      1 | tokenizer.chat_template                          | `{%- macro format_parameters(pr`...`   {%- endif -%}
{%- endif -%}`                                                                                                                                        |
|  44 | BOOL      |      1 | tokenizer.ggml.add_space_prefix                  | False                                                                                                                                                                                                      |
|  45 | BOOL      |      1 | tokenizer.ggml.add_bos_token                     | True                                                                                                                                                                                                       |
|  46 | UINT32    |      1 | general.quantization_version                     | 2                                                                                                                                                                                                          |
|  47 | UINT32    |      1 | general.file_type                                | 7                                                                                                                                                                                                          |

## Tensors Overview ~25B Elements

Total number of elements in all tensors: 25250987068 Elements

- [Base Tensor Group - ~756M Elements](#base)
- [Block 0 Tensor Group - ~814M Elements](#blk_0)
- [Block 1 Tensor Group - ~814M Elements](#blk_1)
- [Block 2 Tensor Group - ~814M Elements](#blk_2)
- [Block 3 Tensor Group - ~814M Elements](#blk_3)
- [Block 4 Tensor Group - ~814M Elements](#blk_4)
- [Block 5 Tensor Group - ~829M Elements](#blk_5)
- [Block 6 Tensor Group - ~814M Elements](#blk_6)
- [Block 7 Tensor Group - ~814M Elements](#blk_7)
- [Block 8 Tensor Group - ~814M Elements](#blk_8)
- [Block 9 Tensor Group - ~814M Elements](#blk_9)
- [Block 10 Tensor Group - ~814M Elements](#blk_10)
- [Block 11 Tensor Group - ~829M Elements](#blk_11)
- [Block 12 Tensor Group - ~814M Elements](#blk_12)
- [Block 13 Tensor Group - ~814M Elements](#blk_13)
- [Block 14 Tensor Group - ~814M Elements](#blk_14)
- [Block 15 Tensor Group - ~814M Elements](#blk_15)
- [Block 16 Tensor Group - ~814M Elements](#blk_16)
- [Block 17 Tensor Group - ~829M Elements](#blk_17)
- [Block 18 Tensor Group - ~814M Elements](#blk_18)
- [Block 19 Tensor Group - ~814M Elements](#blk_19)
- [Block 20 Tensor Group - ~814M Elements](#blk_20)
- [Block 21 Tensor Group - ~814M Elements](#blk_21)
- [Block 22 Tensor Group - ~814M Elements](#blk_22)
- [Block 23 Tensor Group - ~829M Elements](#blk_23)
- [Block 24 Tensor Group - ~814M Elements](#blk_24)
- [Block 25 Tensor Group - ~814M Elements](#blk_25)
- [Block 26 Tensor Group - ~814M Elements](#blk_26)
- [Block 27 Tensor Group - ~814M Elements](#blk_27)
- [Block 28 Tensor Group - ~814M Elements](#blk_28)
- [Block 29 Tensor Group - ~829M Elements](#blk_29)

### Tensor Data Offset

This table contains the offset and data segment relative to start of file

| T_ID | Tensor Layer Name                    |  Data Offset (B) |    Data Size (B) |
|-----:|:-------------------------------------|-----------------:|-----------------:|
|    0 | output_norm.weight                   |         0xf17760 |           0x2c00 |
|    1 | rope_freqs.weight                    |         0xf1a360 |            0x400 |
|    2 | self_cond_down.weight                |         0xf1a760 |         0x606c00 |
|    3 | self_cond_gate.weight                |        0x1521360 |         0x606c00 |
|    4 | self_cond_pre_norm.weight            |        0x1b27f60 |           0x2c00 |
|    5 | self_cond_up.weight                  |        0x1b2ab60 |         0x606c00 |
|    6 | token_embd.weight                    |        0x2131760 |       0x2ec00000 |
|    7 | blk.0.attn_k.weight                  |       0x30d31760 |         0x5d8000 |
|    8 | blk.0.attn_k_norm.weight             |       0x31309760 |            0x400 |
|    9 | blk.0.attn_norm.weight               |       0x31309b60 |           0x2c00 |
|   10 | blk.0.attn_output.weight             |       0x3130c760 |         0xbb0000 |
|   11 | blk.0.attn_q.weight                  |       0x31ebc760 |         0xbb0000 |
|   12 | blk.0.attn_q_norm.weight             |       0x32a6c760 |            0x400 |
|   13 | blk.0.attn_v.weight                  |       0x32a6cb60 |         0x5d8000 |
|   14 | blk.0.enc_layer_output_scale.weight  |       0x33044b60 |              0x4 |
|   15 | blk.0.ffn_down.weight                |       0x33044b80 |         0x606c00 |
|   16 | blk.0.ffn_down_exps.scale            |       0x3364b780 |            0x200 |
|   17 | blk.0.ffn_down_exps.weight           |       0x3364b980 |       0x10120000 |
|   18 | blk.0.ffn_gate.weight                |       0x4376b980 |         0x606c00 |
|   19 | blk.0.ffn_gate_inp.scale             |       0x43d72580 |           0x2c00 |
|   20 | blk.0.ffn_gate_inp.weight            |       0x43d75180 |         0x160000 |
|   21 | blk.0.ffn_gate_up_exps.weight        |       0x43ed5180 |       0x20240000 |
|   22 | blk.0.ffn_norm.weight                |       0x64115180 |           0x2c00 |
|   23 | blk.0.ffn_up.weight                  |       0x64117d80 |         0x606c00 |
|   24 | blk.0.layer_output_scale.weight      |       0x6471e980 |              0x4 |
|   25 | blk.0.post_attention_norm.weight     |       0x6471e9a0 |           0x2c00 |
|   26 | blk.0.post_ffw_norm.weight           |       0x647215a0 |           0x2c00 |
|   27 | blk.0.post_ffw_norm_1.weight         |       0x647241a0 |           0x2c00 |
|   28 | blk.0.post_ffw_norm_2.weight         |       0x64726da0 |           0x2c00 |
|   29 | blk.0.pre_ffw_norm_2.weight          |       0x647299a0 |           0x2c00 |
|   30 | blk.1.attn_k.weight                  |       0x6472c5a0 |         0x5d8000 |
|   31 | blk.1.attn_k_norm.weight             |       0x64d045a0 |            0x400 |
|   32 | blk.1.attn_norm.weight               |       0x64d049a0 |           0x2c00 |
|   33 | blk.1.attn_output.weight             |       0x64d075a0 |         0xbb0000 |
|   34 | blk.1.attn_q.weight                  |       0x658b75a0 |         0xbb0000 |
|   35 | blk.1.attn_q_norm.weight             |       0x664675a0 |            0x400 |
|   36 | blk.1.attn_v.weight                  |       0x664679a0 |         0x5d8000 |
|   37 | blk.1.enc_layer_output_scale.weight  |       0x66a3f9a0 |              0x4 |
|   38 | blk.1.ffn_down.weight                |       0x66a3f9c0 |         0x606c00 |
|   39 | blk.1.ffn_down_exps.scale            |       0x670465c0 |            0x200 |
|   40 | blk.1.ffn_down_exps.weight           |       0x670467c0 |       0x10120000 |
|   41 | blk.1.ffn_gate.weight                |       0x771667c0 |         0x606c00 |
|   42 | blk.1.ffn_gate_inp.scale             |       0x7776d3c0 |           0x2c00 |
|   43 | blk.1.ffn_gate_inp.weight            |       0x7776ffc0 |         0x160000 |
|   44 | blk.1.ffn_gate_up_exps.weight        |       0x778cffc0 |       0x20240000 |
|   45 | blk.1.ffn_norm.weight                |       0x97b0ffc0 |           0x2c00 |
|   46 | blk.1.ffn_up.weight                  |       0x97b12bc0 |         0x606c00 |
|   47 | blk.1.layer_output_scale.weight      |       0x981197c0 |              0x4 |
|   48 | blk.1.post_attention_norm.weight     |       0x981197e0 |           0x2c00 |
|   49 | blk.1.post_ffw_norm.weight           |       0x9811c3e0 |           0x2c00 |
|   50 | blk.1.post_ffw_norm_1.weight         |       0x9811efe0 |           0x2c00 |
|   51 | blk.1.post_ffw_norm_2.weight         |       0x98121be0 |           0x2c00 |
|   52 | blk.1.pre_ffw_norm_2.weight          |       0x981247e0 |           0x2c00 |
|   53 | blk.2.attn_k.weight                  |       0x981273e0 |         0x5d8000 |
|   54 | blk.2.attn_k_norm.weight             |       0x986ff3e0 |            0x400 |
|   55 | blk.2.attn_norm.weight               |       0x986ff7e0 |           0x2c00 |
|   56 | blk.2.attn_output.weight             |       0x987023e0 |         0xbb0000 |
|   57 | blk.2.attn_q.weight                  |       0x992b23e0 |         0xbb0000 |
|   58 | blk.2.attn_q_norm.weight             |       0x99e623e0 |            0x400 |
|   59 | blk.2.attn_v.weight                  |       0x99e627e0 |         0x5d8000 |
|   60 | blk.2.enc_layer_output_scale.weight  |       0x9a43a7e0 |              0x4 |
|   61 | blk.2.ffn_down.weight                |       0x9a43a800 |         0x606c00 |
|   62 | blk.2.ffn_down_exps.scale            |       0x9aa41400 |            0x200 |
|   63 | blk.2.ffn_down_exps.weight           |       0x9aa41600 |       0x10120000 |
|   64 | blk.2.ffn_gate.weight                |       0xaab61600 |         0x606c00 |
|   65 | blk.2.ffn_gate_inp.scale             |       0xab168200 |           0x2c00 |
|   66 | blk.2.ffn_gate_inp.weight            |       0xab16ae00 |         0x160000 |
|   67 | blk.2.ffn_gate_up_exps.weight        |       0xab2cae00 |       0x20240000 |
|   68 | blk.2.ffn_norm.weight                |       0xcb50ae00 |           0x2c00 |
|   69 | blk.2.ffn_up.weight                  |       0xcb50da00 |         0x606c00 |
|   70 | blk.2.layer_output_scale.weight      |       0xcbb14600 |              0x4 |
|   71 | blk.2.post_attention_norm.weight     |       0xcbb14620 |           0x2c00 |
|   72 | blk.2.post_ffw_norm.weight           |       0xcbb17220 |           0x2c00 |
|   73 | blk.2.post_ffw_norm_1.weight         |       0xcbb19e20 |           0x2c00 |
|   74 | blk.2.post_ffw_norm_2.weight         |       0xcbb1ca20 |           0x2c00 |
|   75 | blk.2.pre_ffw_norm_2.weight          |       0xcbb1f620 |           0x2c00 |
|   76 | blk.3.attn_k.weight                  |       0xcbb22220 |         0x5d8000 |
|   77 | blk.3.attn_k_norm.weight             |       0xcc0fa220 |            0x400 |
|   78 | blk.3.attn_norm.weight               |       0xcc0fa620 |           0x2c00 |
|   79 | blk.3.attn_output.weight             |       0xcc0fd220 |         0xbb0000 |
|   80 | blk.3.attn_q.weight                  |       0xcccad220 |         0xbb0000 |
|   81 | blk.3.attn_q_norm.weight             |       0xcd85d220 |            0x400 |
|   82 | blk.3.attn_v.weight                  |       0xcd85d620 |         0x5d8000 |
|   83 | blk.3.enc_layer_output_scale.weight  |       0xcde35620 |              0x4 |
|   84 | blk.3.ffn_down.weight                |       0xcde35640 |         0x606c00 |
|   85 | blk.3.ffn_down_exps.scale            |       0xce43c240 |            0x200 |
|   86 | blk.3.ffn_down_exps.weight           |       0xce43c440 |       0x10120000 |
|   87 | blk.3.ffn_gate.weight                |       0xde55c440 |         0x606c00 |
|   88 | blk.3.ffn_gate_inp.scale             |       0xdeb63040 |           0x2c00 |
|   89 | blk.3.ffn_gate_inp.weight            |       0xdeb65c40 |         0x160000 |
|   90 | blk.3.ffn_gate_up_exps.weight        |       0xdecc5c40 |       0x20240000 |
|   91 | blk.3.ffn_norm.weight                |       0xfef05c40 |           0x2c00 |
|   92 | blk.3.ffn_up.weight                  |       0xfef08840 |         0x606c00 |
|   93 | blk.3.layer_output_scale.weight      |       0xff50f440 |              0x4 |
|   94 | blk.3.post_attention_norm.weight     |       0xff50f460 |           0x2c00 |
|   95 | blk.3.post_ffw_norm.weight           |       0xff512060 |           0x2c00 |
|   96 | blk.3.post_ffw_norm_1.weight         |       0xff514c60 |           0x2c00 |
|   97 | blk.3.post_ffw_norm_2.weight         |       0xff517860 |           0x2c00 |
|   98 | blk.3.pre_ffw_norm_2.weight          |       0xff51a460 |           0x2c00 |
|   99 | blk.4.attn_k.weight                  |       0xff51d060 |         0x5d8000 |
|  100 | blk.4.attn_k_norm.weight             |       0xffaf5060 |            0x400 |
|  101 | blk.4.attn_norm.weight               |       0xffaf5460 |           0x2c00 |
|  102 | blk.4.attn_output.weight             |       0xffaf8060 |         0xbb0000 |
|  103 | blk.4.attn_q.weight                  |      0x1006a8060 |         0xbb0000 |
|  104 | blk.4.attn_q_norm.weight             |      0x101258060 |            0x400 |
|  105 | blk.4.attn_v.weight                  |      0x101258460 |         0x5d8000 |
|  106 | blk.4.enc_layer_output_scale.weight  |      0x101830460 |              0x4 |
|  107 | blk.4.ffn_down.weight                |      0x101830480 |         0x606c00 |
|  108 | blk.4.ffn_down_exps.scale            |      0x101e37080 |            0x200 |
|  109 | blk.4.ffn_down_exps.weight           |      0x101e37280 |       0x10120000 |
|  110 | blk.4.ffn_gate.weight                |      0x111f57280 |         0x606c00 |
|  111 | blk.4.ffn_gate_inp.scale             |      0x11255de80 |           0x2c00 |
|  112 | blk.4.ffn_gate_inp.weight            |      0x112560a80 |         0x160000 |
|  113 | blk.4.ffn_gate_up_exps.weight        |      0x1126c0a80 |       0x20240000 |
|  114 | blk.4.ffn_norm.weight                |      0x132900a80 |           0x2c00 |
|  115 | blk.4.ffn_up.weight                  |      0x132903680 |         0x606c00 |
|  116 | blk.4.layer_output_scale.weight      |      0x132f0a280 |              0x4 |
|  117 | blk.4.post_attention_norm.weight     |      0x132f0a2a0 |           0x2c00 |
|  118 | blk.4.post_ffw_norm.weight           |      0x132f0cea0 |           0x2c00 |
|  119 | blk.4.post_ffw_norm_1.weight         |      0x132f0faa0 |           0x2c00 |
|  120 | blk.4.post_ffw_norm_2.weight         |      0x132f126a0 |           0x2c00 |
|  121 | blk.4.pre_ffw_norm_2.weight          |      0x132f152a0 |           0x2c00 |
|  122 | blk.5.attn_k.weight                  |      0x132f17ea0 |         0x2ec000 |
|  123 | blk.5.attn_k_norm.weight             |      0x133203ea0 |            0x800 |
|  124 | blk.5.attn_norm.weight               |      0x1332046a0 |           0x2c00 |
|  125 | blk.5.attn_output.weight             |      0x1332072a0 |        0x1760000 |
|  126 | blk.5.attn_q.weight                  |      0x1349672a0 |        0x1760000 |
|  127 | blk.5.attn_q_norm.weight             |      0x1360c72a0 |            0x800 |
|  128 | blk.5.enc_layer_output_scale.weight  |      0x1360c7aa0 |              0x4 |
|  129 | blk.5.ffn_down.weight                |      0x1360c7ac0 |         0x606c00 |
|  130 | blk.5.ffn_down_exps.scale            |      0x1366ce6c0 |            0x200 |
|  131 | blk.5.ffn_down_exps.weight           |      0x1366ce8c0 |       0x10120000 |
|  132 | blk.5.ffn_gate.weight                |      0x1467ee8c0 |         0x606c00 |
|  133 | blk.5.ffn_gate_inp.scale             |      0x146df54c0 |           0x2c00 |
|  134 | blk.5.ffn_gate_inp.weight            |      0x146df80c0 |         0x160000 |
|  135 | blk.5.ffn_gate_up_exps.weight        |      0x146f580c0 |       0x20240000 |
|  136 | blk.5.ffn_norm.weight                |      0x1671980c0 |           0x2c00 |
|  137 | blk.5.ffn_up.weight                  |      0x16719acc0 |         0x606c00 |
|  138 | blk.5.layer_output_scale.weight      |      0x1677a18c0 |              0x4 |
|  139 | blk.5.post_attention_norm.weight     |      0x1677a18e0 |           0x2c00 |
|  140 | blk.5.post_ffw_norm.weight           |      0x1677a44e0 |           0x2c00 |
|  141 | blk.5.post_ffw_norm_1.weight         |      0x1677a70e0 |           0x2c00 |
|  142 | blk.5.post_ffw_norm_2.weight         |      0x1677a9ce0 |           0x2c00 |
|  143 | blk.5.pre_ffw_norm_2.weight          |      0x1677ac8e0 |           0x2c00 |
|  144 | blk.6.attn_k.weight                  |      0x1677af4e0 |         0x5d8000 |
|  145 | blk.6.attn_k_norm.weight             |      0x167d874e0 |            0x400 |
|  146 | blk.6.attn_norm.weight               |      0x167d878e0 |           0x2c00 |
|  147 | blk.6.attn_output.weight             |      0x167d8a4e0 |         0xbb0000 |
|  148 | blk.6.attn_q.weight                  |      0x16893a4e0 |         0xbb0000 |
|  149 | blk.6.attn_q_norm.weight             |      0x1694ea4e0 |            0x400 |
|  150 | blk.6.attn_v.weight                  |      0x1694ea8e0 |         0x5d8000 |
|  151 | blk.6.enc_layer_output_scale.weight  |      0x169ac28e0 |              0x4 |
|  152 | blk.6.ffn_down.weight                |      0x169ac2900 |         0x606c00 |
|  153 | blk.6.ffn_down_exps.scale            |      0x16a0c9500 |            0x200 |
|  154 | blk.6.ffn_down_exps.weight           |      0x16a0c9700 |       0x10120000 |
|  155 | blk.6.ffn_gate.weight                |      0x17a1e9700 |         0x606c00 |
|  156 | blk.6.ffn_gate_inp.scale             |      0x17a7f0300 |           0x2c00 |
|  157 | blk.6.ffn_gate_inp.weight            |      0x17a7f2f00 |         0x160000 |
|  158 | blk.6.ffn_gate_up_exps.weight        |      0x17a952f00 |       0x20240000 |
|  159 | blk.6.ffn_norm.weight                |      0x19ab92f00 |           0x2c00 |
|  160 | blk.6.ffn_up.weight                  |      0x19ab95b00 |         0x606c00 |
|  161 | blk.6.layer_output_scale.weight      |      0x19b19c700 |              0x4 |
|  162 | blk.6.post_attention_norm.weight     |      0x19b19c720 |           0x2c00 |
|  163 | blk.6.post_ffw_norm.weight           |      0x19b19f320 |           0x2c00 |
|  164 | blk.6.post_ffw_norm_1.weight         |      0x19b1a1f20 |           0x2c00 |
|  165 | blk.6.post_ffw_norm_2.weight         |      0x19b1a4b20 |           0x2c00 |
|  166 | blk.6.pre_ffw_norm_2.weight          |      0x19b1a7720 |           0x2c00 |
|  167 | blk.7.attn_k.weight                  |      0x19b1aa320 |         0x5d8000 |
|  168 | blk.7.attn_k_norm.weight             |      0x19b782320 |            0x400 |
|  169 | blk.7.attn_norm.weight               |      0x19b782720 |           0x2c00 |
|  170 | blk.7.attn_output.weight             |      0x19b785320 |         0xbb0000 |
|  171 | blk.7.attn_q.weight                  |      0x19c335320 |         0xbb0000 |
|  172 | blk.7.attn_q_norm.weight             |      0x19cee5320 |            0x400 |
|  173 | blk.7.attn_v.weight                  |      0x19cee5720 |         0x5d8000 |
|  174 | blk.7.enc_layer_output_scale.weight  |      0x19d4bd720 |              0x4 |
|  175 | blk.7.ffn_down.weight                |      0x19d4bd740 |         0x606c00 |
|  176 | blk.7.ffn_down_exps.scale            |      0x19dac4340 |            0x200 |
|  177 | blk.7.ffn_down_exps.weight           |      0x19dac4540 |       0x10120000 |
|  178 | blk.7.ffn_gate.weight                |      0x1adbe4540 |         0x606c00 |
|  179 | blk.7.ffn_gate_inp.scale             |      0x1ae1eb140 |           0x2c00 |
|  180 | blk.7.ffn_gate_inp.weight            |      0x1ae1edd40 |         0x160000 |
|  181 | blk.7.ffn_gate_up_exps.weight        |      0x1ae34dd40 |       0x20240000 |
|  182 | blk.7.ffn_norm.weight                |      0x1ce58dd40 |           0x2c00 |
|  183 | blk.7.ffn_up.weight                  |      0x1ce590940 |         0x606c00 |
|  184 | blk.7.layer_output_scale.weight      |      0x1ceb97540 |              0x4 |
|  185 | blk.7.post_attention_norm.weight     |      0x1ceb97560 |           0x2c00 |
|  186 | blk.7.post_ffw_norm.weight           |      0x1ceb9a160 |           0x2c00 |
|  187 | blk.7.post_ffw_norm_1.weight         |      0x1ceb9cd60 |           0x2c00 |
|  188 | blk.7.post_ffw_norm_2.weight         |      0x1ceb9f960 |           0x2c00 |
|  189 | blk.7.pre_ffw_norm_2.weight          |      0x1ceba2560 |           0x2c00 |
|  190 | blk.8.attn_k.weight                  |      0x1ceba5160 |         0x5d8000 |
|  191 | blk.8.attn_k_norm.weight             |      0x1cf17d160 |            0x400 |
|  192 | blk.8.attn_norm.weight               |      0x1cf17d560 |           0x2c00 |
|  193 | blk.8.attn_output.weight             |      0x1cf180160 |         0xbb0000 |
|  194 | blk.8.attn_q.weight                  |      0x1cfd30160 |         0xbb0000 |
|  195 | blk.8.attn_q_norm.weight             |      0x1d08e0160 |            0x400 |
|  196 | blk.8.attn_v.weight                  |      0x1d08e0560 |         0x5d8000 |
|  197 | blk.8.enc_layer_output_scale.weight  |      0x1d0eb8560 |              0x4 |
|  198 | blk.8.ffn_down.weight                |      0x1d0eb8580 |         0x606c00 |
|  199 | blk.8.ffn_down_exps.scale            |      0x1d14bf180 |            0x200 |
|  200 | blk.8.ffn_down_exps.weight           |      0x1d14bf380 |       0x10120000 |
|  201 | blk.8.ffn_gate.weight                |      0x1e15df380 |         0x606c00 |
|  202 | blk.8.ffn_gate_inp.scale             |      0x1e1be5f80 |           0x2c00 |
|  203 | blk.8.ffn_gate_inp.weight            |      0x1e1be8b80 |         0x160000 |
|  204 | blk.8.ffn_gate_up_exps.weight        |      0x1e1d48b80 |       0x20240000 |
|  205 | blk.8.ffn_norm.weight                |      0x201f88b80 |           0x2c00 |
|  206 | blk.8.ffn_up.weight                  |      0x201f8b780 |         0x606c00 |
|  207 | blk.8.layer_output_scale.weight      |      0x202592380 |              0x4 |
|  208 | blk.8.post_attention_norm.weight     |      0x2025923a0 |           0x2c00 |
|  209 | blk.8.post_ffw_norm.weight           |      0x202594fa0 |           0x2c00 |
|  210 | blk.8.post_ffw_norm_1.weight         |      0x202597ba0 |           0x2c00 |
|  211 | blk.8.post_ffw_norm_2.weight         |      0x20259a7a0 |           0x2c00 |
|  212 | blk.8.pre_ffw_norm_2.weight          |      0x20259d3a0 |           0x2c00 |
|  213 | blk.9.attn_k.weight                  |      0x20259ffa0 |         0x5d8000 |
|  214 | blk.9.attn_k_norm.weight             |      0x202b77fa0 |            0x400 |
|  215 | blk.9.attn_norm.weight               |      0x202b783a0 |           0x2c00 |
|  216 | blk.9.attn_output.weight             |      0x202b7afa0 |         0xbb0000 |
|  217 | blk.9.attn_q.weight                  |      0x20372afa0 |         0xbb0000 |
|  218 | blk.9.attn_q_norm.weight             |      0x2042dafa0 |            0x400 |
|  219 | blk.9.attn_v.weight                  |      0x2042db3a0 |         0x5d8000 |
|  220 | blk.9.enc_layer_output_scale.weight  |      0x2048b33a0 |              0x4 |
|  221 | blk.9.ffn_down.weight                |      0x2048b33c0 |         0x606c00 |
|  222 | blk.9.ffn_down_exps.scale            |      0x204eb9fc0 |            0x200 |
|  223 | blk.9.ffn_down_exps.weight           |      0x204eba1c0 |       0x10120000 |
|  224 | blk.9.ffn_gate.weight                |      0x214fda1c0 |         0x606c00 |
|  225 | blk.9.ffn_gate_inp.scale             |      0x2155e0dc0 |           0x2c00 |
|  226 | blk.9.ffn_gate_inp.weight            |      0x2155e39c0 |         0x160000 |
|  227 | blk.9.ffn_gate_up_exps.weight        |      0x2157439c0 |       0x20240000 |
|  228 | blk.9.ffn_norm.weight                |      0x2359839c0 |           0x2c00 |
|  229 | blk.9.ffn_up.weight                  |      0x2359865c0 |         0x606c00 |
|  230 | blk.9.layer_output_scale.weight      |      0x235f8d1c0 |              0x4 |
|  231 | blk.9.post_attention_norm.weight     |      0x235f8d1e0 |           0x2c00 |
|  232 | blk.9.post_ffw_norm.weight           |      0x235f8fde0 |           0x2c00 |
|  233 | blk.9.post_ffw_norm_1.weight         |      0x235f929e0 |           0x2c00 |
|  234 | blk.9.post_ffw_norm_2.weight         |      0x235f955e0 |           0x2c00 |
|  235 | blk.9.pre_ffw_norm_2.weight          |      0x235f981e0 |           0x2c00 |
|  236 | blk.10.attn_k.weight                 |      0x235f9ade0 |         0x5d8000 |
|  237 | blk.10.attn_k_norm.weight            |      0x236572de0 |            0x400 |
|  238 | blk.10.attn_norm.weight              |      0x2365731e0 |           0x2c00 |
|  239 | blk.10.attn_output.weight            |      0x236575de0 |         0xbb0000 |
|  240 | blk.10.attn_q.weight                 |      0x237125de0 |         0xbb0000 |
|  241 | blk.10.attn_q_norm.weight            |      0x237cd5de0 |            0x400 |
|  242 | blk.10.attn_v.weight                 |      0x237cd61e0 |         0x5d8000 |
|  243 | blk.10.enc_layer_output_scale.weight |      0x2382ae1e0 |              0x4 |
|  244 | blk.10.ffn_down.weight               |      0x2382ae200 |         0x606c00 |
|  245 | blk.10.ffn_down_exps.scale           |      0x2388b4e00 |            0x200 |
|  246 | blk.10.ffn_down_exps.weight          |      0x2388b5000 |       0x10120000 |
|  247 | blk.10.ffn_gate.weight               |      0x2489d5000 |         0x606c00 |
|  248 | blk.10.ffn_gate_inp.scale            |      0x248fdbc00 |           0x2c00 |
|  249 | blk.10.ffn_gate_inp.weight           |      0x248fde800 |         0x160000 |
|  250 | blk.10.ffn_gate_up_exps.weight       |      0x24913e800 |       0x20240000 |
|  251 | blk.10.ffn_norm.weight               |      0x26937e800 |           0x2c00 |
|  252 | blk.10.ffn_up.weight                 |      0x269381400 |         0x606c00 |
|  253 | blk.10.layer_output_scale.weight     |      0x269988000 |              0x4 |
|  254 | blk.10.post_attention_norm.weight    |      0x269988020 |           0x2c00 |
|  255 | blk.10.post_ffw_norm.weight          |      0x26998ac20 |           0x2c00 |
|  256 | blk.10.post_ffw_norm_1.weight        |      0x26998d820 |           0x2c00 |
|  257 | blk.10.post_ffw_norm_2.weight        |      0x269990420 |           0x2c00 |
|  258 | blk.10.pre_ffw_norm_2.weight         |      0x269993020 |           0x2c00 |
|  259 | blk.11.attn_k.weight                 |      0x269995c20 |         0x2ec000 |
|  260 | blk.11.attn_k_norm.weight            |      0x269c81c20 |            0x800 |
|  261 | blk.11.attn_norm.weight              |      0x269c82420 |           0x2c00 |
|  262 | blk.11.attn_output.weight            |      0x269c85020 |        0x1760000 |
|  263 | blk.11.attn_q.weight                 |      0x26b3e5020 |        0x1760000 |
|  264 | blk.11.attn_q_norm.weight            |      0x26cb45020 |            0x800 |
|  265 | blk.11.enc_layer_output_scale.weight |      0x26cb45820 |              0x4 |
|  266 | blk.11.ffn_down.weight               |      0x26cb45840 |         0x606c00 |
|  267 | blk.11.ffn_down_exps.scale           |      0x26d14c440 |            0x200 |
|  268 | blk.11.ffn_down_exps.weight          |      0x26d14c640 |       0x10120000 |
|  269 | blk.11.ffn_gate.weight               |      0x27d26c640 |         0x606c00 |
|  270 | blk.11.ffn_gate_inp.scale            |      0x27d873240 |           0x2c00 |
|  271 | blk.11.ffn_gate_inp.weight           |      0x27d875e40 |         0x160000 |
|  272 | blk.11.ffn_gate_up_exps.weight       |      0x27d9d5e40 |       0x20240000 |
|  273 | blk.11.ffn_norm.weight               |      0x29dc15e40 |           0x2c00 |
|  274 | blk.11.ffn_up.weight                 |      0x29dc18a40 |         0x606c00 |
|  275 | blk.11.layer_output_scale.weight     |      0x29e21f640 |              0x4 |
|  276 | blk.11.post_attention_norm.weight    |      0x29e21f660 |           0x2c00 |
|  277 | blk.11.post_ffw_norm.weight          |      0x29e222260 |           0x2c00 |
|  278 | blk.11.post_ffw_norm_1.weight        |      0x29e224e60 |           0x2c00 |
|  279 | blk.11.post_ffw_norm_2.weight        |      0x29e227a60 |           0x2c00 |
|  280 | blk.11.pre_ffw_norm_2.weight         |      0x29e22a660 |           0x2c00 |
|  281 | blk.12.attn_k.weight                 |      0x29e22d260 |         0x5d8000 |
|  282 | blk.12.attn_k_norm.weight            |      0x29e805260 |            0x400 |
|  283 | blk.12.attn_norm.weight              |      0x29e805660 |           0x2c00 |
|  284 | blk.12.attn_output.weight            |      0x29e808260 |         0xbb0000 |
|  285 | blk.12.attn_q.weight                 |      0x29f3b8260 |         0xbb0000 |
|  286 | blk.12.attn_q_norm.weight            |      0x29ff68260 |            0x400 |
|  287 | blk.12.attn_v.weight                 |      0x29ff68660 |         0x5d8000 |
|  288 | blk.12.enc_layer_output_scale.weight |      0x2a0540660 |              0x4 |
|  289 | blk.12.ffn_down.weight               |      0x2a0540680 |         0x606c00 |
|  290 | blk.12.ffn_down_exps.scale           |      0x2a0b47280 |            0x200 |
|  291 | blk.12.ffn_down_exps.weight          |      0x2a0b47480 |       0x10120000 |
|  292 | blk.12.ffn_gate.weight               |      0x2b0c67480 |         0x606c00 |
|  293 | blk.12.ffn_gate_inp.scale            |      0x2b126e080 |           0x2c00 |
|  294 | blk.12.ffn_gate_inp.weight           |      0x2b1270c80 |         0x160000 |
|  295 | blk.12.ffn_gate_up_exps.weight       |      0x2b13d0c80 |       0x20240000 |
|  296 | blk.12.ffn_norm.weight               |      0x2d1610c80 |           0x2c00 |
|  297 | blk.12.ffn_up.weight                 |      0x2d1613880 |         0x606c00 |
|  298 | blk.12.layer_output_scale.weight     |      0x2d1c1a480 |              0x4 |
|  299 | blk.12.post_attention_norm.weight    |      0x2d1c1a4a0 |           0x2c00 |
|  300 | blk.12.post_ffw_norm.weight          |      0x2d1c1d0a0 |           0x2c00 |
|  301 | blk.12.post_ffw_norm_1.weight        |      0x2d1c1fca0 |           0x2c00 |
|  302 | blk.12.post_ffw_norm_2.weight        |      0x2d1c228a0 |           0x2c00 |
|  303 | blk.12.pre_ffw_norm_2.weight         |      0x2d1c254a0 |           0x2c00 |
|  304 | blk.13.attn_k.weight                 |      0x2d1c280a0 |         0x5d8000 |
|  305 | blk.13.attn_k_norm.weight            |      0x2d22000a0 |            0x400 |
|  306 | blk.13.attn_norm.weight              |      0x2d22004a0 |           0x2c00 |
|  307 | blk.13.attn_output.weight            |      0x2d22030a0 |         0xbb0000 |
|  308 | blk.13.attn_q.weight                 |      0x2d2db30a0 |         0xbb0000 |
|  309 | blk.13.attn_q_norm.weight            |      0x2d39630a0 |            0x400 |
|  310 | blk.13.attn_v.weight                 |      0x2d39634a0 |         0x5d8000 |
|  311 | blk.13.enc_layer_output_scale.weight |      0x2d3f3b4a0 |              0x4 |
|  312 | blk.13.ffn_down.weight               |      0x2d3f3b4c0 |         0x606c00 |
|  313 | blk.13.ffn_down_exps.scale           |      0x2d45420c0 |            0x200 |
|  314 | blk.13.ffn_down_exps.weight          |      0x2d45422c0 |       0x10120000 |
|  315 | blk.13.ffn_gate.weight               |      0x2e46622c0 |         0x606c00 |
|  316 | blk.13.ffn_gate_inp.scale            |      0x2e4c68ec0 |           0x2c00 |
|  317 | blk.13.ffn_gate_inp.weight           |      0x2e4c6bac0 |         0x160000 |
|  318 | blk.13.ffn_gate_up_exps.weight       |      0x2e4dcbac0 |       0x20240000 |
|  319 | blk.13.ffn_norm.weight               |      0x30500bac0 |           0x2c00 |
|  320 | blk.13.ffn_up.weight                 |      0x30500e6c0 |         0x606c00 |
|  321 | blk.13.layer_output_scale.weight     |      0x3056152c0 |              0x4 |
|  322 | blk.13.post_attention_norm.weight    |      0x3056152e0 |           0x2c00 |
|  323 | blk.13.post_ffw_norm.weight          |      0x305617ee0 |           0x2c00 |
|  324 | blk.13.post_ffw_norm_1.weight        |      0x30561aae0 |           0x2c00 |
|  325 | blk.13.post_ffw_norm_2.weight        |      0x30561d6e0 |           0x2c00 |
|  326 | blk.13.pre_ffw_norm_2.weight         |      0x3056202e0 |           0x2c00 |
|  327 | blk.14.attn_k.weight                 |      0x305622ee0 |         0x5d8000 |
|  328 | blk.14.attn_k_norm.weight            |      0x305bfaee0 |            0x400 |
|  329 | blk.14.attn_norm.weight              |      0x305bfb2e0 |           0x2c00 |
|  330 | blk.14.attn_output.weight            |      0x305bfdee0 |         0xbb0000 |
|  331 | blk.14.attn_q.weight                 |      0x3067adee0 |         0xbb0000 |
|  332 | blk.14.attn_q_norm.weight            |      0x30735dee0 |            0x400 |
|  333 | blk.14.attn_v.weight                 |      0x30735e2e0 |         0x5d8000 |
|  334 | blk.14.enc_layer_output_scale.weight |      0x3079362e0 |              0x4 |
|  335 | blk.14.ffn_down.weight               |      0x307936300 |         0x606c00 |
|  336 | blk.14.ffn_down_exps.scale           |      0x307f3cf00 |            0x200 |
|  337 | blk.14.ffn_down_exps.weight          |      0x307f3d100 |       0x10120000 |
|  338 | blk.14.ffn_gate.weight               |      0x31805d100 |         0x606c00 |
|  339 | blk.14.ffn_gate_inp.scale            |      0x318663d00 |           0x2c00 |
|  340 | blk.14.ffn_gate_inp.weight           |      0x318666900 |         0x160000 |
|  341 | blk.14.ffn_gate_up_exps.weight       |      0x3187c6900 |       0x20240000 |
|  342 | blk.14.ffn_norm.weight               |      0x338a06900 |           0x2c00 |
|  343 | blk.14.ffn_up.weight                 |      0x338a09500 |         0x606c00 |
|  344 | blk.14.layer_output_scale.weight     |      0x339010100 |              0x4 |
|  345 | blk.14.post_attention_norm.weight    |      0x339010120 |           0x2c00 |
|  346 | blk.14.post_ffw_norm.weight          |      0x339012d20 |           0x2c00 |
|  347 | blk.14.post_ffw_norm_1.weight        |      0x339015920 |           0x2c00 |
|  348 | blk.14.post_ffw_norm_2.weight        |      0x339018520 |           0x2c00 |
|  349 | blk.14.pre_ffw_norm_2.weight         |      0x33901b120 |           0x2c00 |
|  350 | blk.15.attn_k.weight                 |      0x33901dd20 |         0x5d8000 |
|  351 | blk.15.attn_k_norm.weight            |      0x3395f5d20 |            0x400 |
|  352 | blk.15.attn_norm.weight              |      0x3395f6120 |           0x2c00 |
|  353 | blk.15.attn_output.weight            |      0x3395f8d20 |         0xbb0000 |
|  354 | blk.15.attn_q.weight                 |      0x33a1a8d20 |         0xbb0000 |
|  355 | blk.15.attn_q_norm.weight            |      0x33ad58d20 |            0x400 |
|  356 | blk.15.attn_v.weight                 |      0x33ad59120 |         0x5d8000 |
|  357 | blk.15.enc_layer_output_scale.weight |      0x33b331120 |              0x4 |
|  358 | blk.15.ffn_down.weight               |      0x33b331140 |         0x606c00 |
|  359 | blk.15.ffn_down_exps.scale           |      0x33b937d40 |            0x200 |
|  360 | blk.15.ffn_down_exps.weight          |      0x33b937f40 |       0x10120000 |
|  361 | blk.15.ffn_gate.weight               |      0x34ba57f40 |         0x606c00 |
|  362 | blk.15.ffn_gate_inp.scale            |      0x34c05eb40 |           0x2c00 |
|  363 | blk.15.ffn_gate_inp.weight           |      0x34c061740 |         0x160000 |
|  364 | blk.15.ffn_gate_up_exps.weight       |      0x34c1c1740 |       0x20240000 |
|  365 | blk.15.ffn_norm.weight               |      0x36c401740 |           0x2c00 |
|  366 | blk.15.ffn_up.weight                 |      0x36c404340 |         0x606c00 |
|  367 | blk.15.layer_output_scale.weight     |      0x36ca0af40 |              0x4 |
|  368 | blk.15.post_attention_norm.weight    |      0x36ca0af60 |           0x2c00 |
|  369 | blk.15.post_ffw_norm.weight          |      0x36ca0db60 |           0x2c00 |
|  370 | blk.15.post_ffw_norm_1.weight        |      0x36ca10760 |           0x2c00 |
|  371 | blk.15.post_ffw_norm_2.weight        |      0x36ca13360 |           0x2c00 |
|  372 | blk.15.pre_ffw_norm_2.weight         |      0x36ca15f60 |           0x2c00 |
|  373 | blk.16.attn_k.weight                 |      0x36ca18b60 |         0x5d8000 |
|  374 | blk.16.attn_k_norm.weight            |      0x36cff0b60 |            0x400 |
|  375 | blk.16.attn_norm.weight              |      0x36cff0f60 |           0x2c00 |
|  376 | blk.16.attn_output.weight            |      0x36cff3b60 |         0xbb0000 |
|  377 | blk.16.attn_q.weight                 |      0x36dba3b60 |         0xbb0000 |
|  378 | blk.16.attn_q_norm.weight            |      0x36e753b60 |            0x400 |
|  379 | blk.16.attn_v.weight                 |      0x36e753f60 |         0x5d8000 |
|  380 | blk.16.enc_layer_output_scale.weight |      0x36ed2bf60 |              0x4 |
|  381 | blk.16.ffn_down.weight               |      0x36ed2bf80 |         0x606c00 |
|  382 | blk.16.ffn_down_exps.scale           |      0x36f332b80 |            0x200 |
|  383 | blk.16.ffn_down_exps.weight          |      0x36f332d80 |       0x10120000 |
|  384 | blk.16.ffn_gate.weight               |      0x37f452d80 |         0x606c00 |
|  385 | blk.16.ffn_gate_inp.scale            |      0x37fa59980 |           0x2c00 |
|  386 | blk.16.ffn_gate_inp.weight           |      0x37fa5c580 |         0x160000 |
|  387 | blk.16.ffn_gate_up_exps.weight       |      0x37fbbc580 |       0x20240000 |
|  388 | blk.16.ffn_norm.weight               |      0x39fdfc580 |           0x2c00 |
|  389 | blk.16.ffn_up.weight                 |      0x39fdff180 |         0x606c00 |
|  390 | blk.16.layer_output_scale.weight     |      0x3a0405d80 |              0x4 |
|  391 | blk.16.post_attention_norm.weight    |      0x3a0405da0 |           0x2c00 |
|  392 | blk.16.post_ffw_norm.weight          |      0x3a04089a0 |           0x2c00 |
|  393 | blk.16.post_ffw_norm_1.weight        |      0x3a040b5a0 |           0x2c00 |
|  394 | blk.16.post_ffw_norm_2.weight        |      0x3a040e1a0 |           0x2c00 |
|  395 | blk.16.pre_ffw_norm_2.weight         |      0x3a0410da0 |           0x2c00 |
|  396 | blk.17.attn_k.weight                 |      0x3a04139a0 |         0x2ec000 |
|  397 | blk.17.attn_k_norm.weight            |      0x3a06ff9a0 |            0x800 |
|  398 | blk.17.attn_norm.weight              |      0x3a07001a0 |           0x2c00 |
|  399 | blk.17.attn_output.weight            |      0x3a0702da0 |        0x1760000 |
|  400 | blk.17.attn_q.weight                 |      0x3a1e62da0 |        0x1760000 |
|  401 | blk.17.attn_q_norm.weight            |      0x3a35c2da0 |            0x800 |
|  402 | blk.17.enc_layer_output_scale.weight |      0x3a35c35a0 |              0x4 |
|  403 | blk.17.ffn_down.weight               |      0x3a35c35c0 |         0x606c00 |
|  404 | blk.17.ffn_down_exps.scale           |      0x3a3bca1c0 |            0x200 |
|  405 | blk.17.ffn_down_exps.weight          |      0x3a3bca3c0 |       0x10120000 |
|  406 | blk.17.ffn_gate.weight               |      0x3b3cea3c0 |         0x606c00 |
|  407 | blk.17.ffn_gate_inp.scale            |      0x3b42f0fc0 |           0x2c00 |
|  408 | blk.17.ffn_gate_inp.weight           |      0x3b42f3bc0 |         0x160000 |
|  409 | blk.17.ffn_gate_up_exps.weight       |      0x3b4453bc0 |       0x20240000 |
|  410 | blk.17.ffn_norm.weight               |      0x3d4693bc0 |           0x2c00 |
|  411 | blk.17.ffn_up.weight                 |      0x3d46967c0 |         0x606c00 |
|  412 | blk.17.layer_output_scale.weight     |      0x3d4c9d3c0 |              0x4 |
|  413 | blk.17.post_attention_norm.weight    |      0x3d4c9d3e0 |           0x2c00 |
|  414 | blk.17.post_ffw_norm.weight          |      0x3d4c9ffe0 |           0x2c00 |
|  415 | blk.17.post_ffw_norm_1.weight        |      0x3d4ca2be0 |           0x2c00 |
|  416 | blk.17.post_ffw_norm_2.weight        |      0x3d4ca57e0 |           0x2c00 |
|  417 | blk.17.pre_ffw_norm_2.weight         |      0x3d4ca83e0 |           0x2c00 |
|  418 | blk.18.attn_k.weight                 |      0x3d4caafe0 |         0x5d8000 |
|  419 | blk.18.attn_k_norm.weight            |      0x3d5282fe0 |            0x400 |
|  420 | blk.18.attn_norm.weight              |      0x3d52833e0 |           0x2c00 |
|  421 | blk.18.attn_output.weight            |      0x3d5285fe0 |         0xbb0000 |
|  422 | blk.18.attn_q.weight                 |      0x3d5e35fe0 |         0xbb0000 |
|  423 | blk.18.attn_q_norm.weight            |      0x3d69e5fe0 |            0x400 |
|  424 | blk.18.attn_v.weight                 |      0x3d69e63e0 |         0x5d8000 |
|  425 | blk.18.enc_layer_output_scale.weight |      0x3d6fbe3e0 |              0x4 |
|  426 | blk.18.ffn_down.weight               |      0x3d6fbe400 |         0x606c00 |
|  427 | blk.18.ffn_down_exps.scale           |      0x3d75c5000 |            0x200 |
|  428 | blk.18.ffn_down_exps.weight          |      0x3d75c5200 |       0x10120000 |
|  429 | blk.18.ffn_gate.weight               |      0x3e76e5200 |         0x606c00 |
|  430 | blk.18.ffn_gate_inp.scale            |      0x3e7cebe00 |           0x2c00 |
|  431 | blk.18.ffn_gate_inp.weight           |      0x3e7ceea00 |         0x160000 |
|  432 | blk.18.ffn_gate_up_exps.weight       |      0x3e7e4ea00 |       0x20240000 |
|  433 | blk.18.ffn_norm.weight               |      0x40808ea00 |           0x2c00 |
|  434 | blk.18.ffn_up.weight                 |      0x408091600 |         0x606c00 |
|  435 | blk.18.layer_output_scale.weight     |      0x408698200 |              0x4 |
|  436 | blk.18.post_attention_norm.weight    |      0x408698220 |           0x2c00 |
|  437 | blk.18.post_ffw_norm.weight          |      0x40869ae20 |           0x2c00 |
|  438 | blk.18.post_ffw_norm_1.weight        |      0x40869da20 |           0x2c00 |
|  439 | blk.18.post_ffw_norm_2.weight        |      0x4086a0620 |           0x2c00 |
|  440 | blk.18.pre_ffw_norm_2.weight         |      0x4086a3220 |           0x2c00 |
|  441 | blk.19.attn_k.weight                 |      0x4086a5e20 |         0x5d8000 |
|  442 | blk.19.attn_k_norm.weight            |      0x408c7de20 |            0x400 |
|  443 | blk.19.attn_norm.weight              |      0x408c7e220 |           0x2c00 |
|  444 | blk.19.attn_output.weight            |      0x408c80e20 |         0xbb0000 |
|  445 | blk.19.attn_q.weight                 |      0x409830e20 |         0xbb0000 |
|  446 | blk.19.attn_q_norm.weight            |      0x40a3e0e20 |            0x400 |
|  447 | blk.19.attn_v.weight                 |      0x40a3e1220 |         0x5d8000 |
|  448 | blk.19.enc_layer_output_scale.weight |      0x40a9b9220 |              0x4 |
|  449 | blk.19.ffn_down.weight               |      0x40a9b9240 |         0x606c00 |
|  450 | blk.19.ffn_down_exps.scale           |      0x40afbfe40 |            0x200 |
|  451 | blk.19.ffn_down_exps.weight          |      0x40afc0040 |       0x10120000 |
|  452 | blk.19.ffn_gate.weight               |      0x41b0e0040 |         0x606c00 |
|  453 | blk.19.ffn_gate_inp.scale            |      0x41b6e6c40 |           0x2c00 |
|  454 | blk.19.ffn_gate_inp.weight           |      0x41b6e9840 |         0x160000 |
|  455 | blk.19.ffn_gate_up_exps.weight       |      0x41b849840 |       0x20240000 |
|  456 | blk.19.ffn_norm.weight               |      0x43ba89840 |           0x2c00 |
|  457 | blk.19.ffn_up.weight                 |      0x43ba8c440 |         0x606c00 |
|  458 | blk.19.layer_output_scale.weight     |      0x43c093040 |              0x4 |
|  459 | blk.19.post_attention_norm.weight    |      0x43c093060 |           0x2c00 |
|  460 | blk.19.post_ffw_norm.weight          |      0x43c095c60 |           0x2c00 |
|  461 | blk.19.post_ffw_norm_1.weight        |      0x43c098860 |           0x2c00 |
|  462 | blk.19.post_ffw_norm_2.weight        |      0x43c09b460 |           0x2c00 |
|  463 | blk.19.pre_ffw_norm_2.weight         |      0x43c09e060 |           0x2c00 |
|  464 | blk.20.attn_k.weight                 |      0x43c0a0c60 |         0x5d8000 |
|  465 | blk.20.attn_k_norm.weight            |      0x43c678c60 |            0x400 |
|  466 | blk.20.attn_norm.weight              |      0x43c679060 |           0x2c00 |
|  467 | blk.20.attn_output.weight            |      0x43c67bc60 |         0xbb0000 |
|  468 | blk.20.attn_q.weight                 |      0x43d22bc60 |         0xbb0000 |
|  469 | blk.20.attn_q_norm.weight            |      0x43dddbc60 |            0x400 |
|  470 | blk.20.attn_v.weight                 |      0x43dddc060 |         0x5d8000 |
|  471 | blk.20.enc_layer_output_scale.weight |      0x43e3b4060 |              0x4 |
|  472 | blk.20.ffn_down.weight               |      0x43e3b4080 |         0x606c00 |
|  473 | blk.20.ffn_down_exps.scale           |      0x43e9bac80 |            0x200 |
|  474 | blk.20.ffn_down_exps.weight          |      0x43e9bae80 |       0x10120000 |
|  475 | blk.20.ffn_gate.weight               |      0x44eadae80 |         0x606c00 |
|  476 | blk.20.ffn_gate_inp.scale            |      0x44f0e1a80 |           0x2c00 |
|  477 | blk.20.ffn_gate_inp.weight           |      0x44f0e4680 |         0x160000 |
|  478 | blk.20.ffn_gate_up_exps.weight       |      0x44f244680 |       0x20240000 |
|  479 | blk.20.ffn_norm.weight               |      0x46f484680 |           0x2c00 |
|  480 | blk.20.ffn_up.weight                 |      0x46f487280 |         0x606c00 |
|  481 | blk.20.layer_output_scale.weight     |      0x46fa8de80 |              0x4 |
|  482 | blk.20.post_attention_norm.weight    |      0x46fa8dea0 |           0x2c00 |
|  483 | blk.20.post_ffw_norm.weight          |      0x46fa90aa0 |           0x2c00 |
|  484 | blk.20.post_ffw_norm_1.weight        |      0x46fa936a0 |           0x2c00 |
|  485 | blk.20.post_ffw_norm_2.weight        |      0x46fa962a0 |           0x2c00 |
|  486 | blk.20.pre_ffw_norm_2.weight         |      0x46fa98ea0 |           0x2c00 |
|  487 | blk.21.attn_k.weight                 |      0x46fa9baa0 |         0x5d8000 |
|  488 | blk.21.attn_k_norm.weight            |      0x470073aa0 |            0x400 |
|  489 | blk.21.attn_norm.weight              |      0x470073ea0 |           0x2c00 |
|  490 | blk.21.attn_output.weight            |      0x470076aa0 |         0xbb0000 |
|  491 | blk.21.attn_q.weight                 |      0x470c26aa0 |         0xbb0000 |
|  492 | blk.21.attn_q_norm.weight            |      0x4717d6aa0 |            0x400 |
|  493 | blk.21.attn_v.weight                 |      0x4717d6ea0 |         0x5d8000 |
|  494 | blk.21.enc_layer_output_scale.weight |      0x471daeea0 |              0x4 |
|  495 | blk.21.ffn_down.weight               |      0x471daeec0 |         0x606c00 |
|  496 | blk.21.ffn_down_exps.scale           |      0x4723b5ac0 |            0x200 |
|  497 | blk.21.ffn_down_exps.weight          |      0x4723b5cc0 |       0x10120000 |
|  498 | blk.21.ffn_gate.weight               |      0x4824d5cc0 |         0x606c00 |
|  499 | blk.21.ffn_gate_inp.scale            |      0x482adc8c0 |           0x2c00 |
|  500 | blk.21.ffn_gate_inp.weight           |      0x482adf4c0 |         0x160000 |
|  501 | blk.21.ffn_gate_up_exps.weight       |      0x482c3f4c0 |       0x20240000 |
|  502 | blk.21.ffn_norm.weight               |      0x4a2e7f4c0 |           0x2c00 |
|  503 | blk.21.ffn_up.weight                 |      0x4a2e820c0 |         0x606c00 |
|  504 | blk.21.layer_output_scale.weight     |      0x4a3488cc0 |              0x4 |
|  505 | blk.21.post_attention_norm.weight    |      0x4a3488ce0 |           0x2c00 |
|  506 | blk.21.post_ffw_norm.weight          |      0x4a348b8e0 |           0x2c00 |
|  507 | blk.21.post_ffw_norm_1.weight        |      0x4a348e4e0 |           0x2c00 |
|  508 | blk.21.post_ffw_norm_2.weight        |      0x4a34910e0 |           0x2c00 |
|  509 | blk.21.pre_ffw_norm_2.weight         |      0x4a3493ce0 |           0x2c00 |
|  510 | blk.22.attn_k.weight                 |      0x4a34968e0 |         0x5d8000 |
|  511 | blk.22.attn_k_norm.weight            |      0x4a3a6e8e0 |            0x400 |
|  512 | blk.22.attn_norm.weight              |      0x4a3a6ece0 |           0x2c00 |
|  513 | blk.22.attn_output.weight            |      0x4a3a718e0 |         0xbb0000 |
|  514 | blk.22.attn_q.weight                 |      0x4a46218e0 |         0xbb0000 |
|  515 | blk.22.attn_q_norm.weight            |      0x4a51d18e0 |            0x400 |
|  516 | blk.22.attn_v.weight                 |      0x4a51d1ce0 |         0x5d8000 |
|  517 | blk.22.enc_layer_output_scale.weight |      0x4a57a9ce0 |              0x4 |
|  518 | blk.22.ffn_down.weight               |      0x4a57a9d00 |         0x606c00 |
|  519 | blk.22.ffn_down_exps.scale           |      0x4a5db0900 |            0x200 |
|  520 | blk.22.ffn_down_exps.weight          |      0x4a5db0b00 |       0x10120000 |
|  521 | blk.22.ffn_gate.weight               |      0x4b5ed0b00 |         0x606c00 |
|  522 | blk.22.ffn_gate_inp.scale            |      0x4b64d7700 |           0x2c00 |
|  523 | blk.22.ffn_gate_inp.weight           |      0x4b64da300 |         0x160000 |
|  524 | blk.22.ffn_gate_up_exps.weight       |      0x4b663a300 |       0x20240000 |
|  525 | blk.22.ffn_norm.weight               |      0x4d687a300 |           0x2c00 |
|  526 | blk.22.ffn_up.weight                 |      0x4d687cf00 |         0x606c00 |
|  527 | blk.22.layer_output_scale.weight     |      0x4d6e83b00 |              0x4 |
|  528 | blk.22.post_attention_norm.weight    |      0x4d6e83b20 |           0x2c00 |
|  529 | blk.22.post_ffw_norm.weight          |      0x4d6e86720 |           0x2c00 |
|  530 | blk.22.post_ffw_norm_1.weight        |      0x4d6e89320 |           0x2c00 |
|  531 | blk.22.post_ffw_norm_2.weight        |      0x4d6e8bf20 |           0x2c00 |
|  532 | blk.22.pre_ffw_norm_2.weight         |      0x4d6e8eb20 |           0x2c00 |
|  533 | blk.23.attn_k.weight                 |      0x4d6e91720 |         0x2ec000 |
|  534 | blk.23.attn_k_norm.weight            |      0x4d717d720 |            0x800 |
|  535 | blk.23.attn_norm.weight              |      0x4d717df20 |           0x2c00 |
|  536 | blk.23.attn_output.weight            |      0x4d7180b20 |        0x1760000 |
|  537 | blk.23.attn_q.weight                 |      0x4d88e0b20 |        0x1760000 |
|  538 | blk.23.attn_q_norm.weight            |      0x4da040b20 |            0x800 |
|  539 | blk.23.enc_layer_output_scale.weight |      0x4da041320 |              0x4 |
|  540 | blk.23.ffn_down.weight               |      0x4da041340 |         0x606c00 |
|  541 | blk.23.ffn_down_exps.scale           |      0x4da647f40 |            0x200 |
|  542 | blk.23.ffn_down_exps.weight          |      0x4da648140 |       0x10120000 |
|  543 | blk.23.ffn_gate.weight               |      0x4ea768140 |         0x606c00 |
|  544 | blk.23.ffn_gate_inp.scale            |      0x4ead6ed40 |           0x2c00 |
|  545 | blk.23.ffn_gate_inp.weight           |      0x4ead71940 |         0x160000 |
|  546 | blk.23.ffn_gate_up_exps.weight       |      0x4eaed1940 |       0x20240000 |
|  547 | blk.23.ffn_norm.weight               |      0x50b111940 |           0x2c00 |
|  548 | blk.23.ffn_up.weight                 |      0x50b114540 |         0x606c00 |
|  549 | blk.23.layer_output_scale.weight     |      0x50b71b140 |              0x4 |
|  550 | blk.23.post_attention_norm.weight    |      0x50b71b160 |           0x2c00 |
|  551 | blk.23.post_ffw_norm.weight          |      0x50b71dd60 |           0x2c00 |
|  552 | blk.23.post_ffw_norm_1.weight        |      0x50b720960 |           0x2c00 |
|  553 | blk.23.post_ffw_norm_2.weight        |      0x50b723560 |           0x2c00 |
|  554 | blk.23.pre_ffw_norm_2.weight         |      0x50b726160 |           0x2c00 |
|  555 | blk.24.attn_k.weight                 |      0x50b728d60 |         0x5d8000 |
|  556 | blk.24.attn_k_norm.weight            |      0x50bd00d60 |            0x400 |
|  557 | blk.24.attn_norm.weight              |      0x50bd01160 |           0x2c00 |
|  558 | blk.24.attn_output.weight            |      0x50bd03d60 |         0xbb0000 |
|  559 | blk.24.attn_q.weight                 |      0x50c8b3d60 |         0xbb0000 |
|  560 | blk.24.attn_q_norm.weight            |      0x50d463d60 |            0x400 |
|  561 | blk.24.attn_v.weight                 |      0x50d464160 |         0x5d8000 |
|  562 | blk.24.enc_layer_output_scale.weight |      0x50da3c160 |              0x4 |
|  563 | blk.24.ffn_down.weight               |      0x50da3c180 |         0x606c00 |
|  564 | blk.24.ffn_down_exps.scale           |      0x50e042d80 |            0x200 |
|  565 | blk.24.ffn_down_exps.weight          |      0x50e042f80 |       0x10120000 |
|  566 | blk.24.ffn_gate.weight               |      0x51e162f80 |         0x606c00 |
|  567 | blk.24.ffn_gate_inp.scale            |      0x51e769b80 |           0x2c00 |
|  568 | blk.24.ffn_gate_inp.weight           |      0x51e76c780 |         0x160000 |
|  569 | blk.24.ffn_gate_up_exps.weight       |      0x51e8cc780 |       0x20240000 |
|  570 | blk.24.ffn_norm.weight               |      0x53eb0c780 |           0x2c00 |
|  571 | blk.24.ffn_up.weight                 |      0x53eb0f380 |         0x606c00 |
|  572 | blk.24.layer_output_scale.weight     |      0x53f115f80 |              0x4 |
|  573 | blk.24.post_attention_norm.weight    |      0x53f115fa0 |           0x2c00 |
|  574 | blk.24.post_ffw_norm.weight          |      0x53f118ba0 |           0x2c00 |
|  575 | blk.24.post_ffw_norm_1.weight        |      0x53f11b7a0 |           0x2c00 |
|  576 | blk.24.post_ffw_norm_2.weight        |      0x53f11e3a0 |           0x2c00 |
|  577 | blk.24.pre_ffw_norm_2.weight         |      0x53f120fa0 |           0x2c00 |
|  578 | blk.25.attn_k.weight                 |      0x53f123ba0 |         0x5d8000 |
|  579 | blk.25.attn_k_norm.weight            |      0x53f6fbba0 |            0x400 |
|  580 | blk.25.attn_norm.weight              |      0x53f6fbfa0 |           0x2c00 |
|  581 | blk.25.attn_output.weight            |      0x53f6feba0 |         0xbb0000 |
|  582 | blk.25.attn_q.weight                 |      0x5402aeba0 |         0xbb0000 |
|  583 | blk.25.attn_q_norm.weight            |      0x540e5eba0 |            0x400 |
|  584 | blk.25.attn_v.weight                 |      0x540e5efa0 |         0x5d8000 |
|  585 | blk.25.enc_layer_output_scale.weight |      0x541436fa0 |              0x4 |
|  586 | blk.25.ffn_down.weight               |      0x541436fc0 |         0x606c00 |
|  587 | blk.25.ffn_down_exps.scale           |      0x541a3dbc0 |            0x200 |
|  588 | blk.25.ffn_down_exps.weight          |      0x541a3ddc0 |       0x10120000 |
|  589 | blk.25.ffn_gate.weight               |      0x551b5ddc0 |         0x606c00 |
|  590 | blk.25.ffn_gate_inp.scale            |      0x5521649c0 |           0x2c00 |
|  591 | blk.25.ffn_gate_inp.weight           |      0x5521675c0 |         0x160000 |
|  592 | blk.25.ffn_gate_up_exps.weight       |      0x5522c75c0 |       0x20240000 |
|  593 | blk.25.ffn_norm.weight               |      0x5725075c0 |           0x2c00 |
|  594 | blk.25.ffn_up.weight                 |      0x57250a1c0 |         0x606c00 |
|  595 | blk.25.layer_output_scale.weight     |      0x572b10dc0 |              0x4 |
|  596 | blk.25.post_attention_norm.weight    |      0x572b10de0 |           0x2c00 |
|  597 | blk.25.post_ffw_norm.weight          |      0x572b139e0 |           0x2c00 |
|  598 | blk.25.post_ffw_norm_1.weight        |      0x572b165e0 |           0x2c00 |
|  599 | blk.25.post_ffw_norm_2.weight        |      0x572b191e0 |           0x2c00 |
|  600 | blk.25.pre_ffw_norm_2.weight         |      0x572b1bde0 |           0x2c00 |
|  601 | blk.26.attn_k.weight                 |      0x572b1e9e0 |         0x5d8000 |
|  602 | blk.26.attn_k_norm.weight            |      0x5730f69e0 |            0x400 |
|  603 | blk.26.attn_norm.weight              |      0x5730f6de0 |           0x2c00 |
|  604 | blk.26.attn_output.weight            |      0x5730f99e0 |         0xbb0000 |
|  605 | blk.26.attn_q.weight                 |      0x573ca99e0 |         0xbb0000 |
|  606 | blk.26.attn_q_norm.weight            |      0x5748599e0 |            0x400 |
|  607 | blk.26.attn_v.weight                 |      0x574859de0 |         0x5d8000 |
|  608 | blk.26.enc_layer_output_scale.weight |      0x574e31de0 |              0x4 |
|  609 | blk.26.ffn_down.weight               |      0x574e31e00 |         0x606c00 |
|  610 | blk.26.ffn_down_exps.scale           |      0x575438a00 |            0x200 |
|  611 | blk.26.ffn_down_exps.weight          |      0x575438c00 |       0x10120000 |
|  612 | blk.26.ffn_gate.weight               |      0x585558c00 |         0x606c00 |
|  613 | blk.26.ffn_gate_inp.scale            |      0x585b5f800 |           0x2c00 |
|  614 | blk.26.ffn_gate_inp.weight           |      0x585b62400 |         0x160000 |
|  615 | blk.26.ffn_gate_up_exps.weight       |      0x585cc2400 |       0x20240000 |
|  616 | blk.26.ffn_norm.weight               |      0x5a5f02400 |           0x2c00 |
|  617 | blk.26.ffn_up.weight                 |      0x5a5f05000 |         0x606c00 |
|  618 | blk.26.layer_output_scale.weight     |      0x5a650bc00 |              0x4 |
|  619 | blk.26.post_attention_norm.weight    |      0x5a650bc20 |           0x2c00 |
|  620 | blk.26.post_ffw_norm.weight          |      0x5a650e820 |           0x2c00 |
|  621 | blk.26.post_ffw_norm_1.weight        |      0x5a6511420 |           0x2c00 |
|  622 | blk.26.post_ffw_norm_2.weight        |      0x5a6514020 |           0x2c00 |
|  623 | blk.26.pre_ffw_norm_2.weight         |      0x5a6516c20 |           0x2c00 |
|  624 | blk.27.attn_k.weight                 |      0x5a6519820 |         0x5d8000 |
|  625 | blk.27.attn_k_norm.weight            |      0x5a6af1820 |            0x400 |
|  626 | blk.27.attn_norm.weight              |      0x5a6af1c20 |           0x2c00 |
|  627 | blk.27.attn_output.weight            |      0x5a6af4820 |         0xbb0000 |
|  628 | blk.27.attn_q.weight                 |      0x5a76a4820 |         0xbb0000 |
|  629 | blk.27.attn_q_norm.weight            |      0x5a8254820 |            0x400 |
|  630 | blk.27.attn_v.weight                 |      0x5a8254c20 |         0x5d8000 |
|  631 | blk.27.enc_layer_output_scale.weight |      0x5a882cc20 |              0x4 |
|  632 | blk.27.ffn_down.weight               |      0x5a882cc40 |         0x606c00 |
|  633 | blk.27.ffn_down_exps.scale           |      0x5a8e33840 |            0x200 |
|  634 | blk.27.ffn_down_exps.weight          |      0x5a8e33a40 |       0x10120000 |
|  635 | blk.27.ffn_gate.weight               |      0x5b8f53a40 |         0x606c00 |
|  636 | blk.27.ffn_gate_inp.scale            |      0x5b955a640 |           0x2c00 |
|  637 | blk.27.ffn_gate_inp.weight           |      0x5b955d240 |         0x160000 |
|  638 | blk.27.ffn_gate_up_exps.weight       |      0x5b96bd240 |       0x20240000 |
|  639 | blk.27.ffn_norm.weight               |      0x5d98fd240 |           0x2c00 |
|  640 | blk.27.ffn_up.weight                 |      0x5d98ffe40 |         0x606c00 |
|  641 | blk.27.layer_output_scale.weight     |      0x5d9f06a40 |              0x4 |
|  642 | blk.27.post_attention_norm.weight    |      0x5d9f06a60 |           0x2c00 |
|  643 | blk.27.post_ffw_norm.weight          |      0x5d9f09660 |           0x2c00 |
|  644 | blk.27.post_ffw_norm_1.weight        |      0x5d9f0c260 |           0x2c00 |
|  645 | blk.27.post_ffw_norm_2.weight        |      0x5d9f0ee60 |           0x2c00 |
|  646 | blk.27.pre_ffw_norm_2.weight         |      0x5d9f11a60 |           0x2c00 |
|  647 | blk.28.attn_k.weight                 |      0x5d9f14660 |         0x5d8000 |
|  648 | blk.28.attn_k_norm.weight            |      0x5da4ec660 |            0x400 |
|  649 | blk.28.attn_norm.weight              |      0x5da4eca60 |           0x2c00 |
|  650 | blk.28.attn_output.weight            |      0x5da4ef660 |         0xbb0000 |
|  651 | blk.28.attn_q.weight                 |      0x5db09f660 |         0xbb0000 |
|  652 | blk.28.attn_q_norm.weight            |      0x5dbc4f660 |            0x400 |
|  653 | blk.28.attn_v.weight                 |      0x5dbc4fa60 |         0x5d8000 |
|  654 | blk.28.enc_layer_output_scale.weight |      0x5dc227a60 |              0x4 |
|  655 | blk.28.ffn_down.weight               |      0x5dc227a80 |         0x606c00 |
|  656 | blk.28.ffn_down_exps.scale           |      0x5dc82e680 |            0x200 |
|  657 | blk.28.ffn_down_exps.weight          |      0x5dc82e880 |       0x10120000 |
|  658 | blk.28.ffn_gate.weight               |      0x5ec94e880 |         0x606c00 |
|  659 | blk.28.ffn_gate_inp.scale            |      0x5ecf55480 |           0x2c00 |
|  660 | blk.28.ffn_gate_inp.weight           |      0x5ecf58080 |         0x160000 |
|  661 | blk.28.ffn_gate_up_exps.weight       |      0x5ed0b8080 |       0x20240000 |
|  662 | blk.28.ffn_norm.weight               |      0x60d2f8080 |           0x2c00 |
|  663 | blk.28.ffn_up.weight                 |      0x60d2fac80 |         0x606c00 |
|  664 | blk.28.layer_output_scale.weight     |      0x60d901880 |              0x4 |
|  665 | blk.28.post_attention_norm.weight    |      0x60d9018a0 |           0x2c00 |
|  666 | blk.28.post_ffw_norm.weight          |      0x60d9044a0 |           0x2c00 |
|  667 | blk.28.post_ffw_norm_1.weight        |      0x60d9070a0 |           0x2c00 |
|  668 | blk.28.post_ffw_norm_2.weight        |      0x60d909ca0 |           0x2c00 |
|  669 | blk.28.pre_ffw_norm_2.weight         |      0x60d90c8a0 |           0x2c00 |
|  670 | blk.29.attn_k.weight                 |      0x60d90f4a0 |         0x2ec000 |
|  671 | blk.29.attn_k_norm.weight            |      0x60dbfb4a0 |            0x800 |
|  672 | blk.29.attn_norm.weight              |      0x60dbfbca0 |           0x2c00 |
|  673 | blk.29.attn_output.weight            |      0x60dbfe8a0 |        0x1760000 |
|  674 | blk.29.attn_q.weight                 |      0x60f35e8a0 |        0x1760000 |
|  675 | blk.29.attn_q_norm.weight            |      0x610abe8a0 |            0x800 |
|  676 | blk.29.enc_layer_output_scale.weight |      0x610abf0a0 |              0x4 |
|  677 | blk.29.ffn_down.weight               |      0x610abf0c0 |         0x606c00 |
|  678 | blk.29.ffn_down_exps.scale           |      0x6110c5cc0 |            0x200 |
|  679 | blk.29.ffn_down_exps.weight          |      0x6110c5ec0 |       0x10120000 |
|  680 | blk.29.ffn_gate.weight               |      0x6211e5ec0 |         0x606c00 |
|  681 | blk.29.ffn_gate_inp.scale            |      0x6217ecac0 |           0x2c00 |
|  682 | blk.29.ffn_gate_inp.weight           |      0x6217ef6c0 |         0x160000 |
|  683 | blk.29.ffn_gate_up_exps.weight       |      0x62194f6c0 |       0x20240000 |
|  684 | blk.29.ffn_norm.weight               |      0x641b8f6c0 |           0x2c00 |
|  685 | blk.29.ffn_up.weight                 |      0x641b922c0 |         0x606c00 |
|  686 | blk.29.layer_output_scale.weight     |      0x642198ec0 |              0x4 |
|  687 | blk.29.post_attention_norm.weight    |      0x642198ee0 |           0x2c00 |
|  688 | blk.29.post_ffw_norm.weight          |      0x64219bae0 |           0x2c00 |
|  689 | blk.29.post_ffw_norm_1.weight        |      0x64219e6e0 |           0x2c00 |
|  690 | blk.29.post_ffw_norm_2.weight        |      0x6421a12e0 |           0x2c00 |
|  691 | blk.29.pre_ffw_norm_2.weight         |      0x6421a3ee0 |           0x2c00 |

### <a name="base">Base Tensor Group : ~756M Elements</a>

| T_ID | Tensor Layer Name         | Human Friendly Tensor Layer Name | Elements          | Shape                 | Type |     BPW |
|-----:|:--------------------------|:---------------------------------|:------------------|:----------------------|:-----|--------:|
|    0 | output_norm.weight        | Output Normalization (W)         | (  ~3K)      2816 | 2816 x      1 x 1 x 1 | F32  | 32.0000 |
|    1 | rope_freqs.weight         | Rope_Freqs (W)                   | (  256)       256 |  256 x      1 x 1 x 1 | F32  | 32.0000 |
|    2 | self_cond_down.weight     | Self_Cond_Down (W)               | (  ~6M)   5947392 | 2112 x   2816 x 1 x 1 | Q8_0 |  8.5000 |
|    3 | self_cond_gate.weight     | Self_Cond_Gate (W)               | (  ~6M)   5947392 | 2816 x   2112 x 1 x 1 | Q8_0 |  8.5000 |
|    4 | self_cond_pre_norm.weight | Self_Cond_Pre_Norm (W)           | (  ~3K)      2816 | 2816 x      1 x 1 x 1 | F32  | 32.0000 |
|    5 | self_cond_up.weight       | Self_Cond_Up (W)                 | (  ~6M)   5947392 | 2816 x   2112 x 1 x 1 | Q8_0 |  8.5000 |
|    6 | token_embd.weight         | Token Embedding (W)              | (~738M) 738197504 | 2816 x 262144 x 1 x 1 | Q8_0 |  8.5000 |

- Total elements in base: (~756M) 756045568
- Percentage of total elements: 2.99%
- Bits per Weight (BPW) for base: 8.5002 bits


### <a name="blk_0">Block 0 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|    7 | blk.0.attn_k.weight                 | Block 0 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|    8 | blk.0.attn_k_norm.weight            | Block 0 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|    9 | blk.0.attn_norm.weight              | Block 0 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   10 | blk.0.attn_output.weight            | Block 0 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|   11 | blk.0.attn_q.weight                 | Block 0 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|   12 | blk.0.attn_q_norm.weight            | Block 0 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|   13 | blk.0.attn_v.weight                 | Block 0 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|   14 | blk.0.enc_layer_output_scale.weight | Block 0 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|   15 | blk.0.ffn_down.weight               | Block 0 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|   16 | blk.0.ffn_down_exps.scale           | Block 0 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|   17 | blk.0.ffn_down_exps.weight          | Block 0 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|   18 | blk.0.ffn_gate.weight               | Block 0 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|   19 | blk.0.ffn_gate_inp.scale            | Block 0 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   20 | blk.0.ffn_gate_inp.weight           | Block 0 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|   21 | blk.0.ffn_gate_up_exps.weight       | Block 0 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|   22 | blk.0.ffn_norm.weight               | Block 0 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   23 | blk.0.ffn_up.weight                 | Block 0 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|   24 | blk.0.layer_output_scale.weight     | Block 0 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|   25 | blk.0.post_attention_norm.weight    | Block 0 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   26 | blk.0.post_ffw_norm.weight          | Block 0 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   27 | blk.0.post_ffw_norm_1.weight        | Block 0 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   28 | blk.0.post_ffw_norm_2.weight        | Block 0 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   29 | blk.0.pre_ffw_norm_2.weight         | Block 0 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.0: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.0: 8.5111 bits


### <a name="blk_1">Block 1 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|   30 | blk.1.attn_k.weight                 | Block 1 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|   31 | blk.1.attn_k_norm.weight            | Block 1 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|   32 | blk.1.attn_norm.weight              | Block 1 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   33 | blk.1.attn_output.weight            | Block 1 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|   34 | blk.1.attn_q.weight                 | Block 1 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|   35 | blk.1.attn_q_norm.weight            | Block 1 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|   36 | blk.1.attn_v.weight                 | Block 1 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|   37 | blk.1.enc_layer_output_scale.weight | Block 1 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|   38 | blk.1.ffn_down.weight               | Block 1 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|   39 | blk.1.ffn_down_exps.scale           | Block 1 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|   40 | blk.1.ffn_down_exps.weight          | Block 1 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|   41 | blk.1.ffn_gate.weight               | Block 1 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|   42 | blk.1.ffn_gate_inp.scale            | Block 1 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   43 | blk.1.ffn_gate_inp.weight           | Block 1 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|   44 | blk.1.ffn_gate_up_exps.weight       | Block 1 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|   45 | blk.1.ffn_norm.weight               | Block 1 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   46 | blk.1.ffn_up.weight                 | Block 1 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|   47 | blk.1.layer_output_scale.weight     | Block 1 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|   48 | blk.1.post_attention_norm.weight    | Block 1 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   49 | blk.1.post_ffw_norm.weight          | Block 1 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   50 | blk.1.post_ffw_norm_1.weight        | Block 1 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   51 | blk.1.post_ffw_norm_2.weight        | Block 1 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   52 | blk.1.pre_ffw_norm_2.weight         | Block 1 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.1: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.1: 8.5111 bits


### <a name="blk_2">Block 2 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|   53 | blk.2.attn_k.weight                 | Block 2 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|   54 | blk.2.attn_k_norm.weight            | Block 2 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|   55 | blk.2.attn_norm.weight              | Block 2 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   56 | blk.2.attn_output.weight            | Block 2 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|   57 | blk.2.attn_q.weight                 | Block 2 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|   58 | blk.2.attn_q_norm.weight            | Block 2 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|   59 | blk.2.attn_v.weight                 | Block 2 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|   60 | blk.2.enc_layer_output_scale.weight | Block 2 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|   61 | blk.2.ffn_down.weight               | Block 2 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|   62 | blk.2.ffn_down_exps.scale           | Block 2 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|   63 | blk.2.ffn_down_exps.weight          | Block 2 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|   64 | blk.2.ffn_gate.weight               | Block 2 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|   65 | blk.2.ffn_gate_inp.scale            | Block 2 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   66 | blk.2.ffn_gate_inp.weight           | Block 2 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|   67 | blk.2.ffn_gate_up_exps.weight       | Block 2 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|   68 | blk.2.ffn_norm.weight               | Block 2 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   69 | blk.2.ffn_up.weight                 | Block 2 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|   70 | blk.2.layer_output_scale.weight     | Block 2 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|   71 | blk.2.post_attention_norm.weight    | Block 2 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   72 | blk.2.post_ffw_norm.weight          | Block 2 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   73 | blk.2.post_ffw_norm_1.weight        | Block 2 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   74 | blk.2.post_ffw_norm_2.weight        | Block 2 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   75 | blk.2.pre_ffw_norm_2.weight         | Block 2 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.2: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.2: 8.5111 bits


### <a name="blk_3">Block 3 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|   76 | blk.3.attn_k.weight                 | Block 3 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|   77 | blk.3.attn_k_norm.weight            | Block 3 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|   78 | blk.3.attn_norm.weight              | Block 3 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   79 | blk.3.attn_output.weight            | Block 3 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|   80 | blk.3.attn_q.weight                 | Block 3 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|   81 | blk.3.attn_q_norm.weight            | Block 3 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|   82 | blk.3.attn_v.weight                 | Block 3 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|   83 | blk.3.enc_layer_output_scale.weight | Block 3 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|   84 | blk.3.ffn_down.weight               | Block 3 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|   85 | blk.3.ffn_down_exps.scale           | Block 3 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|   86 | blk.3.ffn_down_exps.weight          | Block 3 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|   87 | blk.3.ffn_gate.weight               | Block 3 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|   88 | blk.3.ffn_gate_inp.scale            | Block 3 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   89 | blk.3.ffn_gate_inp.weight           | Block 3 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|   90 | blk.3.ffn_gate_up_exps.weight       | Block 3 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|   91 | blk.3.ffn_norm.weight               | Block 3 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   92 | blk.3.ffn_up.weight                 | Block 3 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|   93 | blk.3.layer_output_scale.weight     | Block 3 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|   94 | blk.3.post_attention_norm.weight    | Block 3 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   95 | blk.3.post_ffw_norm.weight          | Block 3 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   96 | blk.3.post_ffw_norm_1.weight        | Block 3 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   97 | blk.3.post_ffw_norm_2.weight        | Block 3 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|   98 | blk.3.pre_ffw_norm_2.weight         | Block 3 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.3: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.3: 8.5111 bits


### <a name="blk_4">Block 4 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|   99 | blk.4.attn_k.weight                 | Block 4 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  100 | blk.4.attn_k_norm.weight            | Block 4 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  101 | blk.4.attn_norm.weight              | Block 4 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  102 | blk.4.attn_output.weight            | Block 4 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  103 | blk.4.attn_q.weight                 | Block 4 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  104 | blk.4.attn_q_norm.weight            | Block 4 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  105 | blk.4.attn_v.weight                 | Block 4 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  106 | blk.4.enc_layer_output_scale.weight | Block 4 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  107 | blk.4.ffn_down.weight               | Block 4 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  108 | blk.4.ffn_down_exps.scale           | Block 4 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  109 | blk.4.ffn_down_exps.weight          | Block 4 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  110 | blk.4.ffn_gate.weight               | Block 4 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  111 | blk.4.ffn_gate_inp.scale            | Block 4 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  112 | blk.4.ffn_gate_inp.weight           | Block 4 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  113 | blk.4.ffn_gate_up_exps.weight       | Block 4 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  114 | blk.4.ffn_norm.weight               | Block 4 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  115 | blk.4.ffn_up.weight                 | Block 4 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  116 | blk.4.layer_output_scale.weight     | Block 4 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  117 | blk.4.post_attention_norm.weight    | Block 4 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  118 | blk.4.post_ffw_norm.weight          | Block 4 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  119 | blk.4.post_ffw_norm_1.weight        | Block 4 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  120 | blk.4.post_ffw_norm_2.weight        | Block 4 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  121 | blk.4.pre_ffw_norm_2.weight         | Block 4 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.4: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.4: 8.5111 bits


### <a name="blk_5">Block 5 Tensor Group : ~829M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  122 | blk.5.attn_k.weight                 | Block 5 Attention Key (W)                                                                   | (  ~3M)   2883584 | 2816 x 1024 x   1 x 1 | Q8_0 |  8.5000 |
|  123 | blk.5.attn_k_norm.weight            | Block 5 Attn_K_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  124 | blk.5.attn_norm.weight              | Block 5 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  125 | blk.5.attn_output.weight            | Block 5 Attention Output (W)                                                                | ( ~23M)  23068672 | 8192 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  126 | blk.5.attn_q.weight                 | Block 5 Attention Query (W)                                                                 | ( ~23M)  23068672 | 2816 x 8192 x   1 x 1 | Q8_0 |  8.5000 |
|  127 | blk.5.attn_q_norm.weight            | Block 5 Attn_Q_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  128 | blk.5.enc_layer_output_scale.weight | Block 5 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  129 | blk.5.ffn_down.weight               | Block 5 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  130 | blk.5.ffn_down_exps.scale           | Block 5 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  131 | blk.5.ffn_down_exps.weight          | Block 5 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  132 | blk.5.ffn_gate.weight               | Block 5 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  133 | blk.5.ffn_gate_inp.scale            | Block 5 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  134 | blk.5.ffn_gate_inp.weight           | Block 5 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  135 | blk.5.ffn_gate_up_exps.weight       | Block 5 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  136 | blk.5.ffn_norm.weight               | Block 5 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  137 | blk.5.ffn_up.weight                 | Block 5 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  138 | blk.5.layer_output_scale.weight     | Block 5 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  139 | blk.5.post_attention_norm.weight    | Block 5 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  140 | blk.5.post_ffw_norm.weight          | Block 5 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  141 | blk.5.post_ffw_norm_1.weight        | Block 5 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  142 | blk.5.post_ffw_norm_2.weight        | Block 5 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  143 | blk.5.pre_ffw_norm_2.weight         | Block 5 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.5: (~829M) 828513410
- Percentage of total elements: 3.28%
- Bits per Weight (BPW) for blk.5: 8.5109 bits


### <a name="blk_6">Block 6 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  144 | blk.6.attn_k.weight                 | Block 6 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  145 | blk.6.attn_k_norm.weight            | Block 6 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  146 | blk.6.attn_norm.weight              | Block 6 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  147 | blk.6.attn_output.weight            | Block 6 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  148 | blk.6.attn_q.weight                 | Block 6 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  149 | blk.6.attn_q_norm.weight            | Block 6 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  150 | blk.6.attn_v.weight                 | Block 6 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  151 | blk.6.enc_layer_output_scale.weight | Block 6 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  152 | blk.6.ffn_down.weight               | Block 6 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  153 | blk.6.ffn_down_exps.scale           | Block 6 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  154 | blk.6.ffn_down_exps.weight          | Block 6 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  155 | blk.6.ffn_gate.weight               | Block 6 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  156 | blk.6.ffn_gate_inp.scale            | Block 6 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  157 | blk.6.ffn_gate_inp.weight           | Block 6 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  158 | blk.6.ffn_gate_up_exps.weight       | Block 6 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  159 | blk.6.ffn_norm.weight               | Block 6 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  160 | blk.6.ffn_up.weight                 | Block 6 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  161 | blk.6.layer_output_scale.weight     | Block 6 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  162 | blk.6.post_attention_norm.weight    | Block 6 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  163 | blk.6.post_ffw_norm.weight          | Block 6 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  164 | blk.6.post_ffw_norm_1.weight        | Block 6 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  165 | blk.6.post_ffw_norm_2.weight        | Block 6 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  166 | blk.6.pre_ffw_norm_2.weight         | Block 6 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.6: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.6: 8.5111 bits


### <a name="blk_7">Block 7 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  167 | blk.7.attn_k.weight                 | Block 7 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  168 | blk.7.attn_k_norm.weight            | Block 7 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  169 | blk.7.attn_norm.weight              | Block 7 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  170 | blk.7.attn_output.weight            | Block 7 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  171 | blk.7.attn_q.weight                 | Block 7 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  172 | blk.7.attn_q_norm.weight            | Block 7 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  173 | blk.7.attn_v.weight                 | Block 7 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  174 | blk.7.enc_layer_output_scale.weight | Block 7 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  175 | blk.7.ffn_down.weight               | Block 7 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  176 | blk.7.ffn_down_exps.scale           | Block 7 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  177 | blk.7.ffn_down_exps.weight          | Block 7 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  178 | blk.7.ffn_gate.weight               | Block 7 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  179 | blk.7.ffn_gate_inp.scale            | Block 7 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  180 | blk.7.ffn_gate_inp.weight           | Block 7 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  181 | blk.7.ffn_gate_up_exps.weight       | Block 7 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  182 | blk.7.ffn_norm.weight               | Block 7 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  183 | blk.7.ffn_up.weight                 | Block 7 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  184 | blk.7.layer_output_scale.weight     | Block 7 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  185 | blk.7.post_attention_norm.weight    | Block 7 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  186 | blk.7.post_ffw_norm.weight          | Block 7 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  187 | blk.7.post_ffw_norm_1.weight        | Block 7 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  188 | blk.7.post_ffw_norm_2.weight        | Block 7 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  189 | blk.7.pre_ffw_norm_2.weight         | Block 7 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.7: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.7: 8.5111 bits


### <a name="blk_8">Block 8 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  190 | blk.8.attn_k.weight                 | Block 8 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  191 | blk.8.attn_k_norm.weight            | Block 8 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  192 | blk.8.attn_norm.weight              | Block 8 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  193 | blk.8.attn_output.weight            | Block 8 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  194 | blk.8.attn_q.weight                 | Block 8 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  195 | blk.8.attn_q_norm.weight            | Block 8 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  196 | blk.8.attn_v.weight                 | Block 8 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  197 | blk.8.enc_layer_output_scale.weight | Block 8 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  198 | blk.8.ffn_down.weight               | Block 8 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  199 | blk.8.ffn_down_exps.scale           | Block 8 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  200 | blk.8.ffn_down_exps.weight          | Block 8 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  201 | blk.8.ffn_gate.weight               | Block 8 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  202 | blk.8.ffn_gate_inp.scale            | Block 8 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  203 | blk.8.ffn_gate_inp.weight           | Block 8 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  204 | blk.8.ffn_gate_up_exps.weight       | Block 8 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  205 | blk.8.ffn_norm.weight               | Block 8 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  206 | blk.8.ffn_up.weight                 | Block 8 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  207 | blk.8.layer_output_scale.weight     | Block 8 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  208 | blk.8.post_attention_norm.weight    | Block 8 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  209 | blk.8.post_ffw_norm.weight          | Block 8 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  210 | blk.8.post_ffw_norm_1.weight        | Block 8 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  211 | blk.8.post_ffw_norm_2.weight        | Block 8 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  212 | blk.8.pre_ffw_norm_2.weight         | Block 8 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.8: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.8: 8.5111 bits


### <a name="blk_9">Block 9 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                   | Human Friendly Tensor Layer Name                                                            | Elements          | Shape                 | Type |     BPW |
|-----:|:------------------------------------|:--------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  213 | blk.9.attn_k.weight                 | Block 9 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  214 | blk.9.attn_k_norm.weight            | Block 9 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  215 | blk.9.attn_norm.weight              | Block 9 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  216 | blk.9.attn_output.weight            | Block 9 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  217 | blk.9.attn_q.weight                 | Block 9 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  218 | blk.9.attn_q_norm.weight            | Block 9 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  219 | blk.9.attn_v.weight                 | Block 9 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  220 | blk.9.enc_layer_output_scale.weight | Block 9 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  221 | blk.9.ffn_down.weight               | Block 9 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  222 | blk.9.ffn_down_exps.scale           | Block 9 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  223 | blk.9.ffn_down_exps.weight          | Block 9 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  224 | blk.9.ffn_gate.weight               | Block 9 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  225 | blk.9.ffn_gate_inp.scale            | Block 9 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  226 | blk.9.ffn_gate_inp.weight           | Block 9 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  227 | blk.9.ffn_gate_up_exps.weight       | Block 9 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  228 | blk.9.ffn_norm.weight               | Block 9 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  229 | blk.9.ffn_up.weight                 | Block 9 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  230 | blk.9.layer_output_scale.weight     | Block 9 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  231 | blk.9.post_attention_norm.weight    | Block 9 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  232 | blk.9.post_ffw_norm.weight          | Block 9 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  233 | blk.9.post_ffw_norm_1.weight        | Block 9 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  234 | blk.9.post_ffw_norm_2.weight        | Block 9 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  235 | blk.9.pre_ffw_norm_2.weight         | Block 9 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.9: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.9: 8.5111 bits


### <a name="blk_10">Block 10 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  236 | blk.10.attn_k.weight                 | Block 10 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  237 | blk.10.attn_k_norm.weight            | Block 10 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  238 | blk.10.attn_norm.weight              | Block 10 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  239 | blk.10.attn_output.weight            | Block 10 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  240 | blk.10.attn_q.weight                 | Block 10 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  241 | blk.10.attn_q_norm.weight            | Block 10 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  242 | blk.10.attn_v.weight                 | Block 10 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  243 | blk.10.enc_layer_output_scale.weight | Block 10 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  244 | blk.10.ffn_down.weight               | Block 10 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  245 | blk.10.ffn_down_exps.scale           | Block 10 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  246 | blk.10.ffn_down_exps.weight          | Block 10 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  247 | blk.10.ffn_gate.weight               | Block 10 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  248 | blk.10.ffn_gate_inp.scale            | Block 10 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  249 | blk.10.ffn_gate_inp.weight           | Block 10 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  250 | blk.10.ffn_gate_up_exps.weight       | Block 10 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  251 | blk.10.ffn_norm.weight               | Block 10 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  252 | blk.10.ffn_up.weight                 | Block 10 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  253 | blk.10.layer_output_scale.weight     | Block 10 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  254 | blk.10.post_attention_norm.weight    | Block 10 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  255 | blk.10.post_ffw_norm.weight          | Block 10 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  256 | blk.10.post_ffw_norm_1.weight        | Block 10 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  257 | blk.10.post_ffw_norm_2.weight        | Block 10 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  258 | blk.10.pre_ffw_norm_2.weight         | Block 10 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.10: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.10: 8.5111 bits


### <a name="blk_11">Block 11 Tensor Group : ~829M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  259 | blk.11.attn_k.weight                 | Block 11 Attention Key (W)                                                                   | (  ~3M)   2883584 | 2816 x 1024 x   1 x 1 | Q8_0 |  8.5000 |
|  260 | blk.11.attn_k_norm.weight            | Block 11 Attn_K_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  261 | blk.11.attn_norm.weight              | Block 11 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  262 | blk.11.attn_output.weight            | Block 11 Attention Output (W)                                                                | ( ~23M)  23068672 | 8192 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  263 | blk.11.attn_q.weight                 | Block 11 Attention Query (W)                                                                 | ( ~23M)  23068672 | 2816 x 8192 x   1 x 1 | Q8_0 |  8.5000 |
|  264 | blk.11.attn_q_norm.weight            | Block 11 Attn_Q_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  265 | blk.11.enc_layer_output_scale.weight | Block 11 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  266 | blk.11.ffn_down.weight               | Block 11 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  267 | blk.11.ffn_down_exps.scale           | Block 11 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  268 | blk.11.ffn_down_exps.weight          | Block 11 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  269 | blk.11.ffn_gate.weight               | Block 11 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  270 | blk.11.ffn_gate_inp.scale            | Block 11 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  271 | blk.11.ffn_gate_inp.weight           | Block 11 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  272 | blk.11.ffn_gate_up_exps.weight       | Block 11 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  273 | blk.11.ffn_norm.weight               | Block 11 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  274 | blk.11.ffn_up.weight                 | Block 11 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  275 | blk.11.layer_output_scale.weight     | Block 11 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  276 | blk.11.post_attention_norm.weight    | Block 11 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  277 | blk.11.post_ffw_norm.weight          | Block 11 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  278 | blk.11.post_ffw_norm_1.weight        | Block 11 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  279 | blk.11.post_ffw_norm_2.weight        | Block 11 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  280 | blk.11.pre_ffw_norm_2.weight         | Block 11 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.11: (~829M) 828513410
- Percentage of total elements: 3.28%
- Bits per Weight (BPW) for blk.11: 8.5109 bits


### <a name="blk_12">Block 12 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  281 | blk.12.attn_k.weight                 | Block 12 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  282 | blk.12.attn_k_norm.weight            | Block 12 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  283 | blk.12.attn_norm.weight              | Block 12 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  284 | blk.12.attn_output.weight            | Block 12 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  285 | blk.12.attn_q.weight                 | Block 12 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  286 | blk.12.attn_q_norm.weight            | Block 12 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  287 | blk.12.attn_v.weight                 | Block 12 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  288 | blk.12.enc_layer_output_scale.weight | Block 12 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  289 | blk.12.ffn_down.weight               | Block 12 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  290 | blk.12.ffn_down_exps.scale           | Block 12 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  291 | blk.12.ffn_down_exps.weight          | Block 12 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  292 | blk.12.ffn_gate.weight               | Block 12 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  293 | blk.12.ffn_gate_inp.scale            | Block 12 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  294 | blk.12.ffn_gate_inp.weight           | Block 12 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  295 | blk.12.ffn_gate_up_exps.weight       | Block 12 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  296 | blk.12.ffn_norm.weight               | Block 12 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  297 | blk.12.ffn_up.weight                 | Block 12 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  298 | blk.12.layer_output_scale.weight     | Block 12 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  299 | blk.12.post_attention_norm.weight    | Block 12 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  300 | blk.12.post_ffw_norm.weight          | Block 12 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  301 | blk.12.post_ffw_norm_1.weight        | Block 12 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  302 | blk.12.post_ffw_norm_2.weight        | Block 12 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  303 | blk.12.pre_ffw_norm_2.weight         | Block 12 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.12: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.12: 8.5111 bits


### <a name="blk_13">Block 13 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  304 | blk.13.attn_k.weight                 | Block 13 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  305 | blk.13.attn_k_norm.weight            | Block 13 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  306 | blk.13.attn_norm.weight              | Block 13 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  307 | blk.13.attn_output.weight            | Block 13 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  308 | blk.13.attn_q.weight                 | Block 13 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  309 | blk.13.attn_q_norm.weight            | Block 13 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  310 | blk.13.attn_v.weight                 | Block 13 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  311 | blk.13.enc_layer_output_scale.weight | Block 13 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  312 | blk.13.ffn_down.weight               | Block 13 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  313 | blk.13.ffn_down_exps.scale           | Block 13 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  314 | blk.13.ffn_down_exps.weight          | Block 13 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  315 | blk.13.ffn_gate.weight               | Block 13 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  316 | blk.13.ffn_gate_inp.scale            | Block 13 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  317 | blk.13.ffn_gate_inp.weight           | Block 13 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  318 | blk.13.ffn_gate_up_exps.weight       | Block 13 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  319 | blk.13.ffn_norm.weight               | Block 13 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  320 | blk.13.ffn_up.weight                 | Block 13 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  321 | blk.13.layer_output_scale.weight     | Block 13 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  322 | blk.13.post_attention_norm.weight    | Block 13 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  323 | blk.13.post_ffw_norm.weight          | Block 13 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  324 | blk.13.post_ffw_norm_1.weight        | Block 13 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  325 | blk.13.post_ffw_norm_2.weight        | Block 13 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  326 | blk.13.pre_ffw_norm_2.weight         | Block 13 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.13: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.13: 8.5111 bits


### <a name="blk_14">Block 14 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  327 | blk.14.attn_k.weight                 | Block 14 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  328 | blk.14.attn_k_norm.weight            | Block 14 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  329 | blk.14.attn_norm.weight              | Block 14 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  330 | blk.14.attn_output.weight            | Block 14 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  331 | blk.14.attn_q.weight                 | Block 14 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  332 | blk.14.attn_q_norm.weight            | Block 14 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  333 | blk.14.attn_v.weight                 | Block 14 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  334 | blk.14.enc_layer_output_scale.weight | Block 14 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  335 | blk.14.ffn_down.weight               | Block 14 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  336 | blk.14.ffn_down_exps.scale           | Block 14 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  337 | blk.14.ffn_down_exps.weight          | Block 14 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  338 | blk.14.ffn_gate.weight               | Block 14 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  339 | blk.14.ffn_gate_inp.scale            | Block 14 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  340 | blk.14.ffn_gate_inp.weight           | Block 14 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  341 | blk.14.ffn_gate_up_exps.weight       | Block 14 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  342 | blk.14.ffn_norm.weight               | Block 14 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  343 | blk.14.ffn_up.weight                 | Block 14 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  344 | blk.14.layer_output_scale.weight     | Block 14 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  345 | blk.14.post_attention_norm.weight    | Block 14 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  346 | blk.14.post_ffw_norm.weight          | Block 14 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  347 | blk.14.post_ffw_norm_1.weight        | Block 14 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  348 | blk.14.post_ffw_norm_2.weight        | Block 14 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  349 | blk.14.pre_ffw_norm_2.weight         | Block 14 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.14: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.14: 8.5111 bits


### <a name="blk_15">Block 15 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  350 | blk.15.attn_k.weight                 | Block 15 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  351 | blk.15.attn_k_norm.weight            | Block 15 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  352 | blk.15.attn_norm.weight              | Block 15 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  353 | blk.15.attn_output.weight            | Block 15 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  354 | blk.15.attn_q.weight                 | Block 15 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  355 | blk.15.attn_q_norm.weight            | Block 15 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  356 | blk.15.attn_v.weight                 | Block 15 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  357 | blk.15.enc_layer_output_scale.weight | Block 15 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  358 | blk.15.ffn_down.weight               | Block 15 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  359 | blk.15.ffn_down_exps.scale           | Block 15 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  360 | blk.15.ffn_down_exps.weight          | Block 15 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  361 | blk.15.ffn_gate.weight               | Block 15 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  362 | blk.15.ffn_gate_inp.scale            | Block 15 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  363 | blk.15.ffn_gate_inp.weight           | Block 15 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  364 | blk.15.ffn_gate_up_exps.weight       | Block 15 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  365 | blk.15.ffn_norm.weight               | Block 15 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  366 | blk.15.ffn_up.weight                 | Block 15 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  367 | blk.15.layer_output_scale.weight     | Block 15 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  368 | blk.15.post_attention_norm.weight    | Block 15 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  369 | blk.15.post_ffw_norm.weight          | Block 15 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  370 | blk.15.post_ffw_norm_1.weight        | Block 15 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  371 | blk.15.post_ffw_norm_2.weight        | Block 15 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  372 | blk.15.pre_ffw_norm_2.weight         | Block 15 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.15: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.15: 8.5111 bits


### <a name="blk_16">Block 16 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  373 | blk.16.attn_k.weight                 | Block 16 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  374 | blk.16.attn_k_norm.weight            | Block 16 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  375 | blk.16.attn_norm.weight              | Block 16 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  376 | blk.16.attn_output.weight            | Block 16 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  377 | blk.16.attn_q.weight                 | Block 16 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  378 | blk.16.attn_q_norm.weight            | Block 16 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  379 | blk.16.attn_v.weight                 | Block 16 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  380 | blk.16.enc_layer_output_scale.weight | Block 16 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  381 | blk.16.ffn_down.weight               | Block 16 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  382 | blk.16.ffn_down_exps.scale           | Block 16 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  383 | blk.16.ffn_down_exps.weight          | Block 16 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  384 | blk.16.ffn_gate.weight               | Block 16 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  385 | blk.16.ffn_gate_inp.scale            | Block 16 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  386 | blk.16.ffn_gate_inp.weight           | Block 16 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  387 | blk.16.ffn_gate_up_exps.weight       | Block 16 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  388 | blk.16.ffn_norm.weight               | Block 16 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  389 | blk.16.ffn_up.weight                 | Block 16 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  390 | blk.16.layer_output_scale.weight     | Block 16 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  391 | blk.16.post_attention_norm.weight    | Block 16 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  392 | blk.16.post_ffw_norm.weight          | Block 16 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  393 | blk.16.post_ffw_norm_1.weight        | Block 16 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  394 | blk.16.post_ffw_norm_2.weight        | Block 16 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  395 | blk.16.pre_ffw_norm_2.weight         | Block 16 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.16: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.16: 8.5111 bits


### <a name="blk_17">Block 17 Tensor Group : ~829M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  396 | blk.17.attn_k.weight                 | Block 17 Attention Key (W)                                                                   | (  ~3M)   2883584 | 2816 x 1024 x   1 x 1 | Q8_0 |  8.5000 |
|  397 | blk.17.attn_k_norm.weight            | Block 17 Attn_K_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  398 | blk.17.attn_norm.weight              | Block 17 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  399 | blk.17.attn_output.weight            | Block 17 Attention Output (W)                                                                | ( ~23M)  23068672 | 8192 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  400 | blk.17.attn_q.weight                 | Block 17 Attention Query (W)                                                                 | ( ~23M)  23068672 | 2816 x 8192 x   1 x 1 | Q8_0 |  8.5000 |
|  401 | blk.17.attn_q_norm.weight            | Block 17 Attn_Q_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  402 | blk.17.enc_layer_output_scale.weight | Block 17 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  403 | blk.17.ffn_down.weight               | Block 17 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  404 | blk.17.ffn_down_exps.scale           | Block 17 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  405 | blk.17.ffn_down_exps.weight          | Block 17 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  406 | blk.17.ffn_gate.weight               | Block 17 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  407 | blk.17.ffn_gate_inp.scale            | Block 17 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  408 | blk.17.ffn_gate_inp.weight           | Block 17 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  409 | blk.17.ffn_gate_up_exps.weight       | Block 17 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  410 | blk.17.ffn_norm.weight               | Block 17 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  411 | blk.17.ffn_up.weight                 | Block 17 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  412 | blk.17.layer_output_scale.weight     | Block 17 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  413 | blk.17.post_attention_norm.weight    | Block 17 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  414 | blk.17.post_ffw_norm.weight          | Block 17 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  415 | blk.17.post_ffw_norm_1.weight        | Block 17 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  416 | blk.17.post_ffw_norm_2.weight        | Block 17 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  417 | blk.17.pre_ffw_norm_2.weight         | Block 17 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.17: (~829M) 828513410
- Percentage of total elements: 3.28%
- Bits per Weight (BPW) for blk.17: 8.5109 bits


### <a name="blk_18">Block 18 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  418 | blk.18.attn_k.weight                 | Block 18 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  419 | blk.18.attn_k_norm.weight            | Block 18 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  420 | blk.18.attn_norm.weight              | Block 18 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  421 | blk.18.attn_output.weight            | Block 18 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  422 | blk.18.attn_q.weight                 | Block 18 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  423 | blk.18.attn_q_norm.weight            | Block 18 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  424 | blk.18.attn_v.weight                 | Block 18 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  425 | blk.18.enc_layer_output_scale.weight | Block 18 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  426 | blk.18.ffn_down.weight               | Block 18 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  427 | blk.18.ffn_down_exps.scale           | Block 18 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  428 | blk.18.ffn_down_exps.weight          | Block 18 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  429 | blk.18.ffn_gate.weight               | Block 18 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  430 | blk.18.ffn_gate_inp.scale            | Block 18 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  431 | blk.18.ffn_gate_inp.weight           | Block 18 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  432 | blk.18.ffn_gate_up_exps.weight       | Block 18 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  433 | blk.18.ffn_norm.weight               | Block 18 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  434 | blk.18.ffn_up.weight                 | Block 18 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  435 | blk.18.layer_output_scale.weight     | Block 18 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  436 | blk.18.post_attention_norm.weight    | Block 18 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  437 | blk.18.post_ffw_norm.weight          | Block 18 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  438 | blk.18.post_ffw_norm_1.weight        | Block 18 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  439 | blk.18.post_ffw_norm_2.weight        | Block 18 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  440 | blk.18.pre_ffw_norm_2.weight         | Block 18 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.18: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.18: 8.5111 bits


### <a name="blk_19">Block 19 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  441 | blk.19.attn_k.weight                 | Block 19 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  442 | blk.19.attn_k_norm.weight            | Block 19 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  443 | blk.19.attn_norm.weight              | Block 19 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  444 | blk.19.attn_output.weight            | Block 19 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  445 | blk.19.attn_q.weight                 | Block 19 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  446 | blk.19.attn_q_norm.weight            | Block 19 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  447 | blk.19.attn_v.weight                 | Block 19 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  448 | blk.19.enc_layer_output_scale.weight | Block 19 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  449 | blk.19.ffn_down.weight               | Block 19 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  450 | blk.19.ffn_down_exps.scale           | Block 19 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  451 | blk.19.ffn_down_exps.weight          | Block 19 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  452 | blk.19.ffn_gate.weight               | Block 19 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  453 | blk.19.ffn_gate_inp.scale            | Block 19 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  454 | blk.19.ffn_gate_inp.weight           | Block 19 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  455 | blk.19.ffn_gate_up_exps.weight       | Block 19 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  456 | blk.19.ffn_norm.weight               | Block 19 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  457 | blk.19.ffn_up.weight                 | Block 19 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  458 | blk.19.layer_output_scale.weight     | Block 19 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  459 | blk.19.post_attention_norm.weight    | Block 19 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  460 | blk.19.post_ffw_norm.weight          | Block 19 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  461 | blk.19.post_ffw_norm_1.weight        | Block 19 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  462 | blk.19.post_ffw_norm_2.weight        | Block 19 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  463 | blk.19.pre_ffw_norm_2.weight         | Block 19 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.19: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.19: 8.5111 bits


### <a name="blk_20">Block 20 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  464 | blk.20.attn_k.weight                 | Block 20 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  465 | blk.20.attn_k_norm.weight            | Block 20 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  466 | blk.20.attn_norm.weight              | Block 20 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  467 | blk.20.attn_output.weight            | Block 20 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  468 | blk.20.attn_q.weight                 | Block 20 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  469 | blk.20.attn_q_norm.weight            | Block 20 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  470 | blk.20.attn_v.weight                 | Block 20 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  471 | blk.20.enc_layer_output_scale.weight | Block 20 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  472 | blk.20.ffn_down.weight               | Block 20 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  473 | blk.20.ffn_down_exps.scale           | Block 20 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  474 | blk.20.ffn_down_exps.weight          | Block 20 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  475 | blk.20.ffn_gate.weight               | Block 20 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  476 | blk.20.ffn_gate_inp.scale            | Block 20 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  477 | blk.20.ffn_gate_inp.weight           | Block 20 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  478 | blk.20.ffn_gate_up_exps.weight       | Block 20 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  479 | blk.20.ffn_norm.weight               | Block 20 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  480 | blk.20.ffn_up.weight                 | Block 20 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  481 | blk.20.layer_output_scale.weight     | Block 20 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  482 | blk.20.post_attention_norm.weight    | Block 20 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  483 | blk.20.post_ffw_norm.weight          | Block 20 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  484 | blk.20.post_ffw_norm_1.weight        | Block 20 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  485 | blk.20.post_ffw_norm_2.weight        | Block 20 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  486 | blk.20.pre_ffw_norm_2.weight         | Block 20 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.20: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.20: 8.5111 bits


### <a name="blk_21">Block 21 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  487 | blk.21.attn_k.weight                 | Block 21 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  488 | blk.21.attn_k_norm.weight            | Block 21 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  489 | blk.21.attn_norm.weight              | Block 21 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  490 | blk.21.attn_output.weight            | Block 21 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  491 | blk.21.attn_q.weight                 | Block 21 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  492 | blk.21.attn_q_norm.weight            | Block 21 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  493 | blk.21.attn_v.weight                 | Block 21 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  494 | blk.21.enc_layer_output_scale.weight | Block 21 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  495 | blk.21.ffn_down.weight               | Block 21 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  496 | blk.21.ffn_down_exps.scale           | Block 21 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  497 | blk.21.ffn_down_exps.weight          | Block 21 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  498 | blk.21.ffn_gate.weight               | Block 21 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  499 | blk.21.ffn_gate_inp.scale            | Block 21 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  500 | blk.21.ffn_gate_inp.weight           | Block 21 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  501 | blk.21.ffn_gate_up_exps.weight       | Block 21 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  502 | blk.21.ffn_norm.weight               | Block 21 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  503 | blk.21.ffn_up.weight                 | Block 21 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  504 | blk.21.layer_output_scale.weight     | Block 21 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  505 | blk.21.post_attention_norm.weight    | Block 21 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  506 | blk.21.post_ffw_norm.weight          | Block 21 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  507 | blk.21.post_ffw_norm_1.weight        | Block 21 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  508 | blk.21.post_ffw_norm_2.weight        | Block 21 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  509 | blk.21.pre_ffw_norm_2.weight         | Block 21 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.21: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.21: 8.5111 bits


### <a name="blk_22">Block 22 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  510 | blk.22.attn_k.weight                 | Block 22 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  511 | blk.22.attn_k_norm.weight            | Block 22 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  512 | blk.22.attn_norm.weight              | Block 22 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  513 | blk.22.attn_output.weight            | Block 22 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  514 | blk.22.attn_q.weight                 | Block 22 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  515 | blk.22.attn_q_norm.weight            | Block 22 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  516 | blk.22.attn_v.weight                 | Block 22 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  517 | blk.22.enc_layer_output_scale.weight | Block 22 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  518 | blk.22.ffn_down.weight               | Block 22 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  519 | blk.22.ffn_down_exps.scale           | Block 22 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  520 | blk.22.ffn_down_exps.weight          | Block 22 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  521 | blk.22.ffn_gate.weight               | Block 22 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  522 | blk.22.ffn_gate_inp.scale            | Block 22 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  523 | blk.22.ffn_gate_inp.weight           | Block 22 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  524 | blk.22.ffn_gate_up_exps.weight       | Block 22 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  525 | blk.22.ffn_norm.weight               | Block 22 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  526 | blk.22.ffn_up.weight                 | Block 22 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  527 | blk.22.layer_output_scale.weight     | Block 22 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  528 | blk.22.post_attention_norm.weight    | Block 22 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  529 | blk.22.post_ffw_norm.weight          | Block 22 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  530 | blk.22.post_ffw_norm_1.weight        | Block 22 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  531 | blk.22.post_ffw_norm_2.weight        | Block 22 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  532 | blk.22.pre_ffw_norm_2.weight         | Block 22 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.22: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.22: 8.5111 bits


### <a name="blk_23">Block 23 Tensor Group : ~829M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  533 | blk.23.attn_k.weight                 | Block 23 Attention Key (W)                                                                   | (  ~3M)   2883584 | 2816 x 1024 x   1 x 1 | Q8_0 |  8.5000 |
|  534 | blk.23.attn_k_norm.weight            | Block 23 Attn_K_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  535 | blk.23.attn_norm.weight              | Block 23 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  536 | blk.23.attn_output.weight            | Block 23 Attention Output (W)                                                                | ( ~23M)  23068672 | 8192 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  537 | blk.23.attn_q.weight                 | Block 23 Attention Query (W)                                                                 | ( ~23M)  23068672 | 2816 x 8192 x   1 x 1 | Q8_0 |  8.5000 |
|  538 | blk.23.attn_q_norm.weight            | Block 23 Attn_Q_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  539 | blk.23.enc_layer_output_scale.weight | Block 23 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  540 | blk.23.ffn_down.weight               | Block 23 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  541 | blk.23.ffn_down_exps.scale           | Block 23 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  542 | blk.23.ffn_down_exps.weight          | Block 23 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  543 | blk.23.ffn_gate.weight               | Block 23 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  544 | blk.23.ffn_gate_inp.scale            | Block 23 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  545 | blk.23.ffn_gate_inp.weight           | Block 23 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  546 | blk.23.ffn_gate_up_exps.weight       | Block 23 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  547 | blk.23.ffn_norm.weight               | Block 23 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  548 | blk.23.ffn_up.weight                 | Block 23 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  549 | blk.23.layer_output_scale.weight     | Block 23 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  550 | blk.23.post_attention_norm.weight    | Block 23 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  551 | blk.23.post_ffw_norm.weight          | Block 23 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  552 | blk.23.post_ffw_norm_1.weight        | Block 23 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  553 | blk.23.post_ffw_norm_2.weight        | Block 23 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  554 | blk.23.pre_ffw_norm_2.weight         | Block 23 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.23: (~829M) 828513410
- Percentage of total elements: 3.28%
- Bits per Weight (BPW) for blk.23: 8.5109 bits


### <a name="blk_24">Block 24 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  555 | blk.24.attn_k.weight                 | Block 24 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  556 | blk.24.attn_k_norm.weight            | Block 24 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  557 | blk.24.attn_norm.weight              | Block 24 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  558 | blk.24.attn_output.weight            | Block 24 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  559 | blk.24.attn_q.weight                 | Block 24 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  560 | blk.24.attn_q_norm.weight            | Block 24 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  561 | blk.24.attn_v.weight                 | Block 24 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  562 | blk.24.enc_layer_output_scale.weight | Block 24 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  563 | blk.24.ffn_down.weight               | Block 24 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  564 | blk.24.ffn_down_exps.scale           | Block 24 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  565 | blk.24.ffn_down_exps.weight          | Block 24 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  566 | blk.24.ffn_gate.weight               | Block 24 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  567 | blk.24.ffn_gate_inp.scale            | Block 24 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  568 | blk.24.ffn_gate_inp.weight           | Block 24 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  569 | blk.24.ffn_gate_up_exps.weight       | Block 24 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  570 | blk.24.ffn_norm.weight               | Block 24 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  571 | blk.24.ffn_up.weight                 | Block 24 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  572 | blk.24.layer_output_scale.weight     | Block 24 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  573 | blk.24.post_attention_norm.weight    | Block 24 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  574 | blk.24.post_ffw_norm.weight          | Block 24 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  575 | blk.24.post_ffw_norm_1.weight        | Block 24 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  576 | blk.24.post_ffw_norm_2.weight        | Block 24 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  577 | blk.24.pre_ffw_norm_2.weight         | Block 24 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.24: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.24: 8.5111 bits


### <a name="blk_25">Block 25 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  578 | blk.25.attn_k.weight                 | Block 25 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  579 | blk.25.attn_k_norm.weight            | Block 25 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  580 | blk.25.attn_norm.weight              | Block 25 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  581 | blk.25.attn_output.weight            | Block 25 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  582 | blk.25.attn_q.weight                 | Block 25 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  583 | blk.25.attn_q_norm.weight            | Block 25 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  584 | blk.25.attn_v.weight                 | Block 25 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  585 | blk.25.enc_layer_output_scale.weight | Block 25 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  586 | blk.25.ffn_down.weight               | Block 25 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  587 | blk.25.ffn_down_exps.scale           | Block 25 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  588 | blk.25.ffn_down_exps.weight          | Block 25 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  589 | blk.25.ffn_gate.weight               | Block 25 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  590 | blk.25.ffn_gate_inp.scale            | Block 25 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  591 | blk.25.ffn_gate_inp.weight           | Block 25 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  592 | blk.25.ffn_gate_up_exps.weight       | Block 25 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  593 | blk.25.ffn_norm.weight               | Block 25 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  594 | blk.25.ffn_up.weight                 | Block 25 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  595 | blk.25.layer_output_scale.weight     | Block 25 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  596 | blk.25.post_attention_norm.weight    | Block 25 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  597 | blk.25.post_ffw_norm.weight          | Block 25 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  598 | blk.25.post_ffw_norm_1.weight        | Block 25 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  599 | blk.25.post_ffw_norm_2.weight        | Block 25 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  600 | blk.25.pre_ffw_norm_2.weight         | Block 25 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.25: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.25: 8.5111 bits


### <a name="blk_26">Block 26 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  601 | blk.26.attn_k.weight                 | Block 26 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  602 | blk.26.attn_k_norm.weight            | Block 26 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  603 | blk.26.attn_norm.weight              | Block 26 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  604 | blk.26.attn_output.weight            | Block 26 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  605 | blk.26.attn_q.weight                 | Block 26 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  606 | blk.26.attn_q_norm.weight            | Block 26 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  607 | blk.26.attn_v.weight                 | Block 26 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  608 | blk.26.enc_layer_output_scale.weight | Block 26 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  609 | blk.26.ffn_down.weight               | Block 26 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  610 | blk.26.ffn_down_exps.scale           | Block 26 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  611 | blk.26.ffn_down_exps.weight          | Block 26 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  612 | blk.26.ffn_gate.weight               | Block 26 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  613 | blk.26.ffn_gate_inp.scale            | Block 26 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  614 | blk.26.ffn_gate_inp.weight           | Block 26 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  615 | blk.26.ffn_gate_up_exps.weight       | Block 26 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  616 | blk.26.ffn_norm.weight               | Block 26 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  617 | blk.26.ffn_up.weight                 | Block 26 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  618 | blk.26.layer_output_scale.weight     | Block 26 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  619 | blk.26.post_attention_norm.weight    | Block 26 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  620 | blk.26.post_ffw_norm.weight          | Block 26 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  621 | blk.26.post_ffw_norm_1.weight        | Block 26 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  622 | blk.26.post_ffw_norm_2.weight        | Block 26 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  623 | blk.26.pre_ffw_norm_2.weight         | Block 26 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.26: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.26: 8.5111 bits


### <a name="blk_27">Block 27 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  624 | blk.27.attn_k.weight                 | Block 27 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  625 | blk.27.attn_k_norm.weight            | Block 27 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  626 | blk.27.attn_norm.weight              | Block 27 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  627 | blk.27.attn_output.weight            | Block 27 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  628 | blk.27.attn_q.weight                 | Block 27 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  629 | blk.27.attn_q_norm.weight            | Block 27 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  630 | blk.27.attn_v.weight                 | Block 27 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  631 | blk.27.enc_layer_output_scale.weight | Block 27 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  632 | blk.27.ffn_down.weight               | Block 27 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  633 | blk.27.ffn_down_exps.scale           | Block 27 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  634 | blk.27.ffn_down_exps.weight          | Block 27 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  635 | blk.27.ffn_gate.weight               | Block 27 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  636 | blk.27.ffn_gate_inp.scale            | Block 27 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  637 | blk.27.ffn_gate_inp.weight           | Block 27 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  638 | blk.27.ffn_gate_up_exps.weight       | Block 27 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  639 | blk.27.ffn_norm.weight               | Block 27 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  640 | blk.27.ffn_up.weight                 | Block 27 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  641 | blk.27.layer_output_scale.weight     | Block 27 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  642 | blk.27.post_attention_norm.weight    | Block 27 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  643 | blk.27.post_ffw_norm.weight          | Block 27 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  644 | blk.27.post_ffw_norm_1.weight        | Block 27 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  645 | blk.27.post_ffw_norm_2.weight        | Block 27 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  646 | blk.27.pre_ffw_norm_2.weight         | Block 27 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.27: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.27: 8.5111 bits


### <a name="blk_28">Block 28 Tensor Group : ~814M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  647 | blk.28.attn_k.weight                 | Block 28 Attention Key (W)                                                                   | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  648 | blk.28.attn_k_norm.weight            | Block 28 Attn_K_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  649 | blk.28.attn_norm.weight              | Block 28 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  650 | blk.28.attn_output.weight            | Block 28 Attention Output (W)                                                                | ( ~12M)  11534336 | 4096 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  651 | blk.28.attn_q.weight                 | Block 28 Attention Query (W)                                                                 | ( ~12M)  11534336 | 2816 x 4096 x   1 x 1 | Q8_0 |  8.5000 |
|  652 | blk.28.attn_q_norm.weight            | Block 28 Attn_Q_Norm (W)                                                                     | (  256)       256 |  256 x    1 x   1 x 1 | F32  | 32.0000 |
|  653 | blk.28.attn_v.weight                 | Block 28 Attention Value (W)                                                                 | (  ~6M)   5767168 | 2816 x 2048 x   1 x 1 | Q8_0 |  8.5000 |
|  654 | blk.28.enc_layer_output_scale.weight | Block 28 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  655 | blk.28.ffn_down.weight               | Block 28 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  656 | blk.28.ffn_down_exps.scale           | Block 28 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  657 | blk.28.ffn_down_exps.weight          | Block 28 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  658 | blk.28.ffn_gate.weight               | Block 28 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  659 | blk.28.ffn_gate_inp.scale            | Block 28 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  660 | blk.28.ffn_gate_inp.weight           | Block 28 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  661 | blk.28.ffn_gate_up_exps.weight       | Block 28 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  662 | blk.28.ffn_norm.weight               | Block 28 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  663 | blk.28.ffn_up.weight                 | Block 28 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  664 | blk.28.layer_output_scale.weight     | Block 28 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  665 | blk.28.post_attention_norm.weight    | Block 28 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  666 | blk.28.post_ffw_norm.weight          | Block 28 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  667 | blk.28.post_ffw_norm_1.weight        | Block 28 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  668 | blk.28.post_ffw_norm_2.weight        | Block 28 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  669 | blk.28.pre_ffw_norm_2.weight         | Block 28 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.28: (~814M) 814094978
- Percentage of total elements: 3.22%
- Bits per Weight (BPW) for blk.28: 8.5111 bits


### <a name="blk_29">Block 29 Tensor Group : ~829M Elements</a>

| T_ID | Tensor Layer Name                    | Human Friendly Tensor Layer Name                                                             | Elements          | Shape                 | Type |     BPW |
|-----:|:-------------------------------------|:---------------------------------------------------------------------------------------------|:------------------|:----------------------|:-----|--------:|
|  670 | blk.29.attn_k.weight                 | Block 29 Attention Key (W)                                                                   | (  ~3M)   2883584 | 2816 x 1024 x   1 x 1 | Q8_0 |  8.5000 |
|  671 | blk.29.attn_k_norm.weight            | Block 29 Attn_K_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  672 | blk.29.attn_norm.weight              | Block 29 Attention Normalization (W)                                                         | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  673 | blk.29.attn_output.weight            | Block 29 Attention Output (W)                                                                | ( ~23M)  23068672 | 8192 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  674 | blk.29.attn_q.weight                 | Block 29 Attention Query (W)                                                                 | ( ~23M)  23068672 | 2816 x 8192 x   1 x 1 | Q8_0 |  8.5000 |
|  675 | blk.29.attn_q_norm.weight            | Block 29 Attn_Q_Norm (W)                                                                     | (  512)       512 |  512 x    1 x   1 x 1 | F32  | 32.0000 |
|  676 | blk.29.enc_layer_output_scale.weight | Block 29 Enc_Layer_Output_Scale (W)                                                          | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  677 | blk.29.ffn_down.weight               | Block 29 Feed-Forward Network "Down" (W)                                                     | (  ~6M)   5947392 | 2112 x 2816 x   1 x 1 | Q8_0 |  8.5000 |
|  678 | blk.29.ffn_down_exps.scale           | Block 29 Ffn_Down_Exps Scale                                                                 | (  128)       128 |  128 x    1 x   1 x 1 | F32  | 32.0000 |
|  679 | blk.29.ffn_down_exps.weight          | Block 29 Ffn_Down_Exps (W)                                                                   | (~254M) 253755392 |  704 x 2816 x 128 x 1 | Q8_0 |  8.5000 |
|  680 | blk.29.ffn_gate.weight               | Block 29 Feed-Forward Network "Gate" (W)                                                     | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  681 | blk.29.ffn_gate_inp.scale            | Block 29 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models Scale | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  682 | blk.29.ffn_gate_inp.weight           | Block 29 Expert-Routing Layer For The Feed-Forward Network In Mixture Of Expert Models (W)   | (~360K)    360448 | 2816 x  128 x   1 x 1 | F32  | 32.0000 |
|  683 | blk.29.ffn_gate_up_exps.weight       | Block 29 Ffn_Gate_Up_Exps (W)                                                                | (~508M) 507510784 | 2816 x 1408 x 128 x 1 | Q8_0 |  8.5000 |
|  684 | blk.29.ffn_norm.weight               | Block 29 Feed-Forward Network Normalization (W)                                              | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  685 | blk.29.ffn_up.weight                 | Block 29 Feed-Forward Network "Up" (W)                                                       | (  ~6M)   5947392 | 2816 x 2112 x   1 x 1 | Q8_0 |  8.5000 |
|  686 | blk.29.layer_output_scale.weight     | Block 29 Layer_Output_Scale (W)                                                              | (    1)         1 |    1 x    1 x   1 x 1 | F32  | 32.0000 |
|  687 | blk.29.post_attention_norm.weight    | Block 29 Post_Attention_Norm (W)                                                             | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  688 | blk.29.post_ffw_norm.weight          | Block 29 Post_Ffw_Norm (W)                                                                   | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  689 | blk.29.post_ffw_norm_1.weight        | Block 29 Post_Ffw_Norm_1 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  690 | blk.29.post_ffw_norm_2.weight        | Block 29 Post_Ffw_Norm_2 (W)                                                                 | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |
|  691 | blk.29.pre_ffw_norm_2.weight         | Block 29 Pre_Ffw_Norm_2 (W)                                                                  | (  ~3K)      2816 | 2816 x    1 x   1 x 1 | F32  | 32.0000 |

- Total elements in blk.29: (~829M) 828513410
- Percentage of total elements: 3.28%
- Bits per Weight (BPW) for blk.29: 8.5109 bits


Total BPW for diffusiongemma-26B-A4B-it-Q8_0.gguf: 8.5107 bits
