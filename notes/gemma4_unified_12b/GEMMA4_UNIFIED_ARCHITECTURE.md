# Gemma4 Unified: Encoder-Free Omni-Modal Architecture

**Source**: HuggingFace Transformers (2026)  
**Reference Model**: `google/gemma-4-12B-it`  
**Status**: Cutting-edge; implementation completed 2026

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Configuration System](#configuration-system)
4. [Text Decoder (Language Model)](#text-decoder-language-model)
5. [Vision Pipeline](#vision-pipeline)
6. [Audio Pipeline](#audio-pipeline)
7. [Model Integration & Forward Pass](#model-integration--forward-pass)
8. [Input Preprocessing Requirements](#input-preprocessing-requirements)
9. [Special Features & Optimizations](#special-features--optimizations)
10. [Implementation Guide for Other Languages](#implementation-guide-for-other-languages)

---

## Executive Summary

**Gemma4 Unified** is a decoder-only, omni-modal language model that unifies text, image, video, and audio in a single 262k-token-context transformer. Unlike typical vision-language models (e.g., LLaVA, GPT-4V), it has **no dedicated vision tower and no audio encoder**. Instead:

- **Vision inputs** → lightweight patch embedder (LN → Dense → LN → factorized 2D posemb → LN → RMSNorm → Linear)
- **Audio inputs** → direct raw-waveform projector (no mel-spectrogram, no conformer; just frame chunking → RMSNorm → Linear)
- **All modalities** → merged into text embeddings via masked_scatter, then processed by the shared LM decoder

This design achieves lower latency, tighter memory coupling, and more unified information flow compared to tower-based architectures.

**Key dimensions (12B variant)**:
| Component | Dim | Count | Notes |
|-----------|-----|-------|-------|
| LM vocab | — | 262,144 | SentencePiece |
| LM hidden | 3,840 | — | |
| LM layers | 48 | mix of 5:1 sliding:full | |
| Attn heads | 16 | — | Q only; 8 KV heads |
| KV head dim | 256 | 8 heads | 4 per head |
| Global attn head dim | 512 | 1 head | (for full-attn layers) |
| Vision patch dim | 16 | px | before merge |
| Vision merged patch dim | 48 | px | 3×3 pooling |
| Vision embed | 3,840 | channels | matches LM hidden |
| Audio frame dim | 640 | samples | 40ms @ 16kHz |
| Context length | 262,144 | tokens | |

---

## Architecture Overview

### High-Level Data Flow

```
[Raw Input]
    ↓
[Patchify + Processor] ← Image: 16px patches
                      ← Video: 16px patches per frame
                      ← Audio: 640-sample frames
    ↓
[Modality Embedder] ← Vision: patch_ln1 → dense → patch_ln2 → +pos_emb → multimodal_proj
                    ← Audio:  multimodal_proj (direct projection)
    ↓
[Placeholder Masking & Scatter]
    ↓
[Shared Text Embeddings] (merged with image/audio/video tokens)
    ↓
[Hybrid Attention Transformer] ← 30× [5 sliding (1024) + 1 full (262k)]
                              ← Bidirectional vision attention
                              ← Causal text attention
                              ← Per-layer KV sharing (optional)
    ↓
[RMSNorm → LM Head]
    ↓
[Logits → Loss / Generation]
```

### Class Hierarchy

```
PreTrainedModel
  └─ Gemma4UnifiedPreTrainedModel
      ├─ Gemma4UnifiedTextModel (LlamaModel-based, text-only)
      ├─ Gemma4UnifiedForCausalLM (text LM head)
      ├─ Gemma4UnifiedModel (multimodal trunk)
      └─ Gemma4UnifiedForConditionalGeneration (multimodal top-level)

ProcessorMixin
  └─ Gemma4UnifiedProcessor
      ├─ Gemma4UnifiedImageProcessor (image → pixel_values + position_ids)
      ├─ Gemma4UnifiedVideoProcessor (video → pixel_values_videos + position_ids)
      ├─ Gemma4UnifiedAudioFeatureExtractor (audio → input_features + mask)
      ├─ GemmaTokenizer (text)
      └─ Chat template handling
```

---

## Configuration System

### Composite Config: `Gemma4UnifiedConfig`

The top-level config nests **three sub-configs**, one per modality:

```python
class Gemma4UnifiedConfig(Gemma4Config):
    text_config: Gemma4UnifiedTextConfig | dict
    vision_config: Gemma4UnifiedVisionConfig | dict
    audio_config: Gemma4UnifiedAudioConfig | dict
    
    # Special modality token IDs
    boi_token_id: int = 255_999      # begin-of-image
    eoi_token_id: int = 258_882      # end-of-image
    image_token_id: int = 258_880    # image placeholder
    
    boa_token_id: int = 256_000      # begin-of-audio
    eoa_token_index: int = 258_883   # end-of-audio
    audio_token_id: int = 258_881    # audio placeholder
    
    video_token_id: int = 258_884    # video placeholder
```

#### 1. `Gemma4UnifiedTextConfig`

Extends `Gemma4TextConfig` with hybrid attention scheduling:

```python
@strict
class Gemma4UnifiedTextConfig(PreTrainedConfig):
    # Transformer dimensions
    vocab_size: int = 262_144
    hidden_size: int = 3840
    intermediate_size: int = 9216  # (usually 4× hidden for FFN)
    num_hidden_layers: int = 30    # or 48 for 12B
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    
    # Activation and normalization
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    
    # Position embeddings & context
    max_position_embeddings: int = 262_144
    rope_parameters: dict | None = None  # per-layer-type RoPE config
    
    # Attention scheduling & features
    sliding_window: int = 1024
    layer_types: list[str] | None = None  # "sliding_attention" or "full_attention"
    use_bidirectional_attention: Literal["all", "vision"] = "vision"
    
    # Global attention (full layers)
    num_global_key_value_heads: int | None = None
    global_head_dim: int = 512
    
    # KV sharing & MLP variants
    attention_k_eq_v: bool = False              # K=V projection reuse
    num_kv_shared_layers: int = 0               # consecutive layers sharing KV
    use_double_wide_mlp: bool = False           # fused gate+up
    
    # Final logit processing
    final_logit_softcapping: float | None = None
    
    # Token IDs
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    
    # Special: PLE/MoE not used
    vocab_size_per_layer_input = AttributeError()
    hidden_size_per_layer_input = AttributeError()
    enable_moe_block = AttributeError()
```

**`__post_init__` logic** (critical):
- If `layer_types` is None, auto-generates 5:1 sliding:full pattern:
  ```python
  sliding_window_pattern = 6  # every 6th layer is full_attention
  layer_types = [
      "sliding_attention" if (i + 1) % 6 else "full_attention"
      for i in range(num_hidden_layers)
  ]
  ```
- Forces last layer to `full_attention`
- Sets per-layer RoPE parameters:
  ```python
  rope_parameters = {
      "sliding_attention": {
          "rope_type": "default",
          "rope_theta": 10_000.0
      },
      "full_attention": {
          "rope_type": "proportional",
          "partial_rotary_factor": 0.25,
          "rope_theta": 1_000_000.0  # much larger base
      }
  }
  ```

**Bidirectional attention behavior**:
- `"vision"` (default): vision tokens attend bidirectionally; text tokens use causal masking
- `"all"`: all tokens bidirectional; sliding window halved: `sliding_window = (1024 // 2) + 1 = 513`

#### 2. `Gemma4UnifiedVisionConfig`

Handles image/video embedding configuration:

```python
@strict
class Gemma4UnifiedVisionConfig(PreTrainedConfig):
    # Patch hierarchy
    patch_size: int = 16            # raw image patchification
    pooling_kernel_size: int = 3    # k×k teacher→model patch merge
    
    # Embedding dimensions
    mm_embed_dim: int = 3840        # hidden dim after patch dense projection
    mm_posemb_size: int = 1120      # factorized 2D positional embedding table size
    output_proj_dims: int = 3840    # final projection to text hidden_size
    
    # Normalization
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Derived property (validated)
    @property
    def model_patch_size(self) -> int:
        """Size of merged patches in pixels (48 for 16×3)"""
        return self.patch_size * self.pooling_kernel_size
```

**Patch merging flow**:
- **Teacher patches**: 16×16 px = 256 raw pixels × 3 channels = 768 dims
- **Model patches** (after 3×3 merge): 48×48 px = 2304 raw pixels × 3 channels = 6912 dims
- **Soft tokens** (after Dense proj): 3840 dims (= text hidden size)

#### 3. `Gemma4UnifiedAudioConfig`

Minimalist audio configuration:

```python
@strict
class Gemma4UnifiedAudioConfig(PreTrainedConfig):
    # Raw waveform frame dimension
    audio_embed_dim: int = 640      # samples per audio token (40ms @ 16kHz)
    
    # Normalization
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    @property
    def hidden_size(self) -> int:
        return self.audio_embed_dim
    
    @property
    def output_proj_dims(self) -> int:
        return self.audio_embed_dim
    
    @property
    def audio_samples_per_token(self) -> int:
        return self.audio_embed_dim
```

---

## Text Decoder (Language Model)

### `Gemma4UnifiedTextModel`

Base transformer stack (inherits from `LlamaModel`), specialized for unified architecture:

```python
class Gemma4UnifiedTextModel(Gemma4UnifiedPreTrainedModel, LlamaModel):
    config: Gemma4UnifiedTextConfig
    input_modalities = ("text",)
    
    def __init__(self, config: Gemma4UnifiedTextConfig):
        self.embed_tokens = Gemma4UnifiedTextScaledWordEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            embed_scale=config.hidden_size ** 0.5
        )
        self.norm = Gemma4UnifiedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4UnifiedTextRotaryEmbedding(config)  # per-layer-type RoPE
        
        # 30 or 48 decoder layers
        self.layers = nn.ModuleList([
            Gemma4UnifiedTextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
```

### Attention Mechanism: Hybrid Sliding + Full

#### Layer-Specific RoPE

Each decoder layer has a `layer_type` attribute:

```python
class Gemma4UnifiedTextAttention(nn.Module):
    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):
        self.layer_type = config.layer_types[layer_idx]  # "sliding_attention" or "full_attention"
        self.head_dim = (
            config.global_head_dim if not self.is_sliding and config.global_head_dim
            else config.head_dim
        )
        # RoPE params loaded from rotary_emb based on layer_type
```

#### Sliding Window Attention (5:1 pattern)

For layers with `layer_types[i] == "sliding_attention"`:
- Window size: 1024 tokens
- Query attends to [max(0, position - 1024), position + 1]
- Memory-efficient, O(n) complexity
- Uses `create_sliding_window_causal_mask`

#### Full Attention (every 6th layer)

For layers with `layer_types[i] == "full_attention"`:
- Attends to all previous tokens (causal)
- Head dimension: 512 (vs. 256 for sliding)
- Uses `proportional` RoPE with θ = 1M (vs. 10k for sliding)
- Last layer **always** full attention

#### Bidirectional Vision Attention

When `use_bidirectional_attention == "vision"`:
- Vision tokens (identified by `block_sequence_ids`) attend **both directions**
- Text tokens attend only backwards (causal)
- Implemented via `create_masks_for_generate(..., block_sequence_ids=...)`
- Vision tokens "see" future text; text doesn't see future text

### KV Sharing (Optional)

For `num_kv_shared_layers > 0`:
- Last N layers **reuse** K, V projections from earlier layers
- Reduces parameters & memory
- `shared_kv_states` is a `UserDict` threaded through forward pass (FSDP2-compatible)

```python
class Gemma4UnifiedTextAttention(nn.Module):
    first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
    is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx
    
    if not is_kv_shared_layer:
        # Normal: has k_proj, v_proj
        self.k_proj = nn.Linear(...)
        self.v_proj = nn.Linear(...)
    # else: reuses from earlier layers via shared_kv_states dict
```

### Per-Layer Learnable Scaling

Each decoder layer has a **scalar multiplier** applied to its output:

```python
class Gemma4UnifiedTextDecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config, layer_idx):
        self.register_buffer("layer_scalar", torch.ones(1))  # trainable scalar
    
    def forward(self, hidden_states, ...):
        # ... attention + FFN ...
        hidden_states *= self.layer_scalar  # scale before residual
        return hidden_states
```

Initialized to 1; learned during training for stability.

---

## Vision Pipeline

### Design Principle: Encoder-Free

**Core insight**: Replace expensive vision tower (SigLIP/ViT) with learnable projection pipeline.

**Pipeline**:
```
[Raw Image: (C, H, W)]
    ↓
[Aspect-ratio-preserving resize]
    ↓
[Patchify to 16×16 px]  → (num_teacher_patches, 768)
    ↓
[patches_merge (3×3)]   → (num_model_patches, 6912)
                           with spatial position tracking (x, y)
    ↓
[Embedding: LN → Dense → LN]
    ↓
[Add Factorized 2D Positional Embedding]
    ↓
[LN → Multimodal Embedder]
    ↓
[Final RMSNorm → Linear projection]
    ↓
[Multimodal Hidden States: (num_patches, 3840)]
```

### Step-by-Step Implementation

#### 1. Image Processor: `Gemma4UnifiedImageProcessor`

```python
class Gemma4UnifiedImageProcessor(TorchvisionBackend):
    patch_size: int = 16
    max_soft_tokens: int = 280         # 70, 140, 280, 560, 1120
    pooling_kernel_size: int = 3
    do_resize: bool = True
    do_rescale: bool = True
    do_normalize: bool = False  # No normalization
    image_mean: list = [0.0, 0.0, 0.0]
    image_std: list = [1.0, 1.0, 1.0]
```

**Input**: `images: list[PIL.Image]`  
**Output**: `BatchFeature` with keys:
- `pixel_values`: (batch, max_soft_tokens, 6912)  — merged patch features
- `image_position_ids`: (batch, max_soft_tokens, 2)  — (x, y) coordinates
- `num_soft_tokens_per_image`: list[int]  — actual #tokens per image

#### 2. Aspect-Ratio-Preserving Resize

**Goal**: Fit image within patch budget while preserving aspect ratio.

```python
def get_aspect_ratio_preserving_size(
    height: int, width: int,
    patch_size: int = 16,
    max_patches: int = 252 * 9,  # max_soft_tokens=280 * 3²
    pooling_kernel_size: int = 3,
) -> tuple[int, int]:
    """
    Returns largest (target_h, target_w) such that:
    1. target_h × target_w ≤ max_patches × patch_size²
    2. target_h % (patch_size × pooling_kernel_size) == 0
    3. target_w % (patch_size × pooling_kernel_size) == 0
    """
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = sqrt(target_px / total_px)
    
    # Scale to target
    ideal_h = factor * height
    ideal_w = factor * width
    
    # Quantize to nearest multiple of 48 (16 * 3)
    side_mult = patch_size * pooling_kernel_size
    target_h = floor(ideal_h / side_mult) * side_mult
    target_w = floor(ideal_w / side_mult) * side_mult
    
    # Handle edge cases...
    return target_h, target_w
```

**Example**: 1024×768 image with max_soft_tokens=280 (max_patches=2520):
- `target_px = 2520 × 16² = 645,120`
- `factor = sqrt(645,120 / (1024×768)) ≈ 0.95`
- `ideal_h ≈ 730, ideal_w ≈ 970`
- `target_h = floor(730/48)*48 = 672, target_w = floor(970/48)*48 = 960`
- Final: **672×960** (14×20 = 280 model patches exactly)

#### 3. Patchification

Convert image → flat patch sequence:

```python
def convert_image_to_patches(image: Tensor, patch_size: int = 16) -> Tensor:
    """
    (C, H, W) → (H//patch_size * W//patch_size, patch_size² × C)
    """
    C, H, W = image.shape
    patched = image.reshape(
        C,
        H // patch_size, patch_size,
        W // patch_size, patch_size
    )
    patched = patched.permute(1, 3, 2, 4, 0)  # (ph, pw, ps, ps, C)
    return patched.reshape(H*W // (patch_size**2), patch_size**2 * C)
    # → (280, 768) for 280 model patches
```

**Output**: Teacher patches of 768 dims (16×16 px).

#### 4. Spatial Position Tracking

For each patch, store (x, y) coordinates:

```python
patch_grid = torch.meshgrid(
    torch.arange(patch_width),   # X axis
    torch.arange(patch_height),  # Y axis
    indexing="xy"
)
teacher_positions = torch.stack(patch_grid, dim=-1).reshape(-1, 2)
# → (num_patches, 2) with valid coords in [0, patch_width) × [0, patch_height)
```

#### 5. Patches Merge: 3×3 Pooling with Spatial Grouping

**Goal**: Combine 3×3 teacher patches into 1 model patch, tracking new position.

```python
def patches_merge(
    patches: Tensor,        # (L, 768)
    positions_xy: Tensor,   # (L, 2)
    length: int,            # target length (e.g., 280)
) -> tuple[Tensor, Tensor]:
    """
    Merges k×k spatially-adjacent patches.
    k = sqrt(L / length) = sqrt(9) = 3
    
    Returns:
        merged_patches: (length, 6912)  — 9 × 768
        merged_positions: (length, 2)   — merged position (x//3, y//3)
    """
    patch_size = isqrt(patches.shape[-1] // 3)  # = 16
    k = isqrt(patches.shape[-2] // length)      # = 3
    
    # Step 1: Compute spatial grouping
    # For each patch at (x, y), assign to kernel group (x//3, y//3)
    kernel_idxs = torch.div(positions_xy, k, rounding_mode="floor")
    
    # Step 2: Compute target ordering to group spatially-adjacent patches
    # Ensures patches within each 3×3 kernel are contiguous
    max_x = positions_xy[..., 0].max() + 1
    num_patches_from_top_left = (
        k * k * kernel_idxs[..., 0] +
        k * max_x * kernel_idxs[..., 1]
    )
    position_within_kernel = torch.remainder(positions_xy, k)
    num_patches_from_top_left_of_kernel = (
        position_within_kernel[..., 0] +
        position_within_kernel[..., 1] * k
    )
    target_ordering = (
        num_patches_from_top_left_of_kernel +
        num_patches_from_top_left
    )
    
    # Step 3: Permute patches into kernel-grouped order via argsort
    perm = target_ordering.long().argsort()  # inverse permutation
    kernel_ordered_patches = patches[perm]   # gather by perm
    
    # Step 4: Reshape: (L, 768) → (length, 9, 16, 16, 3)
    #  → (length, 3, 16, 3, 16, 3) via permute
    #  → (length, 6912)
    kernel_ordered_patches = kernel_ordered_patches.reshape(
        length, 9, 16, 16, 3  # (k*k, patch_size, patch_size, channels)
    )
    kernel_ordered_patches = kernel_ordered_patches.reshape(
        length, 3, 3, 16, 16, 3  # (k, k, patch_size, patch_size, channels)
    )
    kernel_ordered_patches = kernel_ordered_patches.permute(
        0, 1, 3, 2, 4, 5  # rearrange spatial dims
    )  # → (length, 48, 48, 3)
    merged_patches = kernel_ordered_patches.reshape(length, 6912)
    
    # Step 5: Update positions (take min x, y within each kernel)
    kernel_ordered_positions = positions_xy[perm]
    kernel_ordered_positions = kernel_ordered_positions.reshape(length, 9, 2)
    new_positions = torch.div(kernel_ordered_positions, k, rounding_mode="floor").min(dim=1)[0]
    
    return merged_patches, new_positions  # (length, 6912), (length, 2)
```

**Key insight**: Merging respects spatial locality — nearby teacher patches become nearby in the model patch.

#### 6. Padding to Fixed Length

```python
def pad_along_first_dim(
    image: Tensor,      # (num_patches, 6912)
    positions: Tensor,  # (num_patches, 2)
    target_length: int = 280,
) -> tuple[Tensor, Tensor]:
    """Pad with zeros (image) and -1 (positions)"""
    pad_len = target_length - image.shape[0]
    if pad_len > 0:
        image = F.pad(image, (0, 0, 0, pad_len))
        positions = F.pad(positions, (0, 0, 0, pad_len), value=-1)
    return image, positions
```

#### 7. Vision Embedder: Lightweight Neural Network

```python
class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Pipeline: LN → Dense → LN → +posemb → LN → RMSNorm → Linear"""
    
    def __init__(self, vision_config, text_config):
        patch_dim = 48 * 48 * 3  # 6912
        mm_dim = vision_config.mm_embed_dim  # 3840
        
        # Patch embedding (LN → Dense → LN)
        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, mm_dim)
        self.patch_ln2 = nn.LayerNorm(mm_dim)
        
        # Factorized 2D positional embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(vision_config.mm_posemb_size, 2, mm_dim)
        )  # shape: (1120, 2, 3840)
        self.pos_norm = nn.LayerNorm(mm_dim)
        
        # Final multimodal projection (RMSNorm → Linear)
        self.multimodal_embedder = Gemma4UnifiedMultimodalEmbedder(
            vision_config, text_config
        )
    
    def forward(
        self,
        pixel_values: Tensor,        # (batch, max_soft_tokens, 6912)
        image_position_ids: Tensor,  # (batch, max_soft_tokens, 2)
    ) -> Tensor:
        """Returns (batch × num_valid_patches, text_hidden_size)"""
        
        # Step 1: Patch embedding (LN → Dense → LN)
        x = self.patch_ln1(pixel_values.float())  # cast to fp32 for LayerNorm
        x = self.patch_dense(x)
        x = self.patch_ln2(x)
        
        # Step 2: Add factorized 2D positional embeddings
        # For each token, look up pos_embedding[(x, y), :, :]
        clamped = image_position_ids.clamp(min=0).long()
        valid = (image_position_ids != -1).float().unsqueeze(-1)  # (batch, tokens, 2, 1)
        axes = torch.arange(2, device=image_position_ids.device)
        
        # pos_embedding[clamped, axes] → gather from (1120, 2, 3840)
        # clamped: (batch, tokens, 2)
        # axes: (2,)
        # Result: (batch, tokens, 2, 3840)
        pos_embs = self.pos_embedding[clamped, axes]  # (batch, tokens, 2, 3840)
        pos_embs = (pos_embs * valid).sum(-2)  # zero out invalid (padding) patches
        
        x = x + pos_embs
        x = self.pos_norm(x)
        
        # Step 3: Final multimodal embedder (RMSNorm → Linear)
        x = self.multimodal_embedder(x)
        
        # Step 4: Strip padding (patches with position_ids == -1)
        padding_mask = (image_position_ids == -1).all(dim=-1)  # (batch, tokens)
        x = x[~padding_mask]  # flatten & remove padding
        
        return x  # (total_valid_patches, text_hidden_size)
```

#### 8. Multimodal Embedder

```python
class Gemma4UnifiedMultimodalEmbedder(Gemma4MultimodalEmbedder):
    """Final projection: RMSNorm → Linear"""
    
    def __init__(self, multimodal_config, text_config):
        self.embedding_pre_projection_norm = Gemma4UnifiedRMSNorm(
            multimodal_config.output_proj_dims,
            eps=multimodal_config.rms_norm_eps
        )
        self.embedding_projection = nn.Linear(
            multimodal_config.output_proj_dims,
            text_config.hidden_size  # 3840
        )
    
    def forward(self, inputs_embeds: Tensor) -> Tensor:
        x = inputs_embeds.to(self.embedding_projection.weight.dtype)
        x = self.embedding_pre_projection_norm(x)  # RMSNorm
        x = self.embedding_projection(x)  # Linear
        return x
```

---

## Audio Pipeline

### Design Principle: Encoder-Free, Raw Waveform

**Core insight**: Skip mel-spectrogram and conformer encoder. Work directly with raw audio frames.

**Pipeline**:
```
[Raw Audio: numpy array, 16 kHz, 1D]
    ↓
[Chunk into 640-sample frames]  (40 ms @ 16 kHz)
    ↓
[Create boolean mask (valid/padding)]
    ↓
[Reshape to (num_frames, 640)]
    ↓
[RMSNorm → Linear projection]  (via multimodal_embedder)
    ↓
[Multimodal Hidden States: (num_frames, 3840)]
```

### Step-by-Step Implementation

#### 1. Audio Feature Extractor: `Gemma4UnifiedAudioFeatureExtractor`

```python
class Gemma4UnifiedAudioFeatureExtractor(SequenceFeatureExtractor):
    """
    Chunks raw waveform into fixed-length frames.
    No mel-spectrogram, no FFT.
    """
    
    feature_size: int = 640          # samples per frame
    sampling_rate: int = 16_000      # Hz
    audio_samples_per_token: int = 640
    padding_value: float = 0.0
    
    model_input_names = ["input_features", "input_features_mask"]
    
    def _extract_waveform_features(
        self,
        waveform: np.ndarray,  # 1D array of samples
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Chunk waveform into (num_tokens, 640) frames.
        Pad to multiple of 640 with zeros.
        
        Returns:
            features: (num_tokens, 640) float32
            mask: (num_tokens,) bool, all True
        """
        # Pad to nearest multiple of 640
        pad_len = (-len(waveform)) % self.audio_samples_per_token
        if pad_len:
            waveform = np.pad(waveform, (0, pad_len))
        
        num_tokens = len(waveform) // self.audio_samples_per_token
        features = waveform.reshape(num_tokens, self.audio_samples_per_token).astype(np.float32)
        mask = np.ones(num_tokens, dtype=bool)  # all valid
        
        return features, mask
    
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray],
        padding: str = "longest",      # "longest" or "max_length"
        max_length: int | None = None,
        truncation: bool = True,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            raw_speech: single waveform or list of waveforms
            padding: strategy for batch padding
            max_length: max frames per audio (optional)
            truncation: whether to truncate oversized audio
        
        Returns BatchFeature with:
            input_features: (batch, max_frames, 640)
            input_features_mask: (batch, max_frames) bool
        """
        # Normalize input to list of 1D arrays
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        elif not isinstance(raw_speech, (list, tuple)):
            raw_speech = [np.asarray(raw_speech)]
        else:
            raw_speech = [np.asarray(s) for s in raw_speech]
        
        # Extract frames for each waveform
        all_features = [
            {"input_features": self._extract_waveform_features(waveform)[0]}
            for waveform in raw_speech
        ]
        
        # Delegate padding/truncation to parent
        padded_inputs = self.pad(
            all_features,
            padding=padding,
            max_length=max_length,
            truncation=truncation and max_length is not None,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        
        # Rename attention_mask → input_features_mask, ensure boolean
        mask = padded_inputs.pop("attention_mask")
        if isinstance(mask, torch.Tensor):
            mask = mask.bool()
        else:
            mask = np.asarray(mask, dtype=bool)
        padded_inputs["input_features_mask"] = mask
        
        return padded_inputs
```

**Example**: 5-second audio @ 16 kHz
- Samples: 5 × 16,000 = 80,000
- Frames: 80,000 / 640 = 125 tokens
- Shape: (125, 640)

#### 2. Audio Embedding in Model

```python
class Gemma4UnifiedModel(Gemma4Model):
    def __init__(self, config: Gemma4UnifiedConfig):
        # ...
        self.embed_audio = (
            Gemma4UnifiedMultimodalEmbedder(config.audio_config, config.text_config)
            if config.audio_config is not None
            else None
        )
    
    @auto_docstring
    def get_audio_features(
        self,
        input_features: Tensor,        # (batch, num_frames, 640)
        input_features_mask: Tensor,   # (batch, num_frames) bool
        **kwargs,
    ) -> Gemma4UnifiedAudioModelOutput:
        """Projects raw waveform frames into text space."""
        if self.embed_audio is None:
            raise ValueError("Audio config not initialized")
        
        # Apply multimodal embedder (RMSNorm → Linear)
        audio_outputs = self.embed_audio(inputs_embeds=input_features)
        # → (batch, num_frames, 3840)
        
        return Gemma4UnifiedAudioModelOutput(
            pooler_output=audio_outputs,
            attention_mask=input_features_mask,
        )
```

---

## Model Integration & Forward Pass

### Class: `Gemma4UnifiedModel`

Top-level multimodal model that merges all modalities into a single text embedding space.

```python
class Gemma4UnifiedModel(Gemma4Model):
    """Encoder-free multimodal trunk (no vision/audio towers)."""
    
    def __init__(self, config: Gemma4UnifiedConfig):
        super().__init__(config)
        del self.audio_tower
        del self.vision_tower
        
        # Vision embedder (encoder-free pipeline)
        self.embed_vision = (
            Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config)
            if config.vision_config is not None
            else None
        )
        
        # Audio embedder (direct projection)
        self.embed_audio = (
            Gemma4UnifiedMultimodalEmbedder(config.audio_config, config.text_config)
            if config.audio_config is not None
            else None
        )
        
        # Base language model (inherited from parent)
        self.language_model = Gemma4UnifiedTextModel(config.text_config)
```

### Forward Pass: `Gemma4UnifiedModel.forward(...)`

```python
def forward(
    self,
    input_ids: LongTensor | None = None,
    pixel_values: FloatTensor | None = None,           # images
    pixel_values_videos: FloatTensor | None = None,    # videos
    input_features: FloatTensor | None = None,         # audio frames
    attention_mask: Tensor | None = None,
    input_features_mask: Tensor | None = None,         # audio validity
    position_ids: LongTensor | None = None,
    past_key_values: Cache | None = None,
    image_position_ids: LongTensor | None = None,      # (batch, max_patches, 2)
    video_position_ids: LongTensor | None = None,
    mm_token_type_ids: LongTensor | None = None,       # block_sequence_ids for vision bidirectional attn
    **kwargs,
) -> Gemma4UnifiedModelOutputWithPast:
    """
    1. Identify modality placeholders in text
    2. Embed each modality
    3. Scatter modality embeddings into placeholder slots
    4. Run shared language model
    5. Return logits + hidden states
    """
    
    # === STEP 1: Parse modality mask ===
    image_mask, video_mask, audio_mask = self.get_placeholder_mask(
        input_ids, inputs_embeds
    )
    # → boolean masks identifying [image_token], [video_token], [audio_token] positions
    
    multimodal_mask = image_mask | video_mask | audio_mask
    
    # === STEP 2: Embed text ===
    llm_input_ids = input_ids.clone() if inputs_embeds is None else None
    if inputs_embeds is None:
        llm_input_ids = torch.where(
            multimodal_mask,
            self.config.text_config.pad_token_id,
            input_ids
        )  # Replace modality tokens with PAD to avoid OOV
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)
    
    # === STEP 3: Embed & scatter images ===
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values, image_position_ids
        ).pooler_output  # (total_valid_patches, 3840)
        
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        
        # Verify token count matches
        n_image_tokens = image_mask.sum()
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        torch_compilable_check(
            inputs_embeds[image_mask_expanded].numel() == image_features.numel(),
            "Image features and image tokens mismatch"
        )
        
        # Scatter: replace [image_token] placeholders with actual embeddings
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask_expanded.to(inputs_embeds.device),
            image_features.to(inputs_embeds.device)
        )
    
    # === STEP 4: Embed & scatter videos ===
    if pixel_values_videos is not None:
        # Flatten frames: (num_videos, num_frames, max_patches, ...) 
        #              → (num_videos × num_frames, max_patches, ...)
        pixel_values_videos_flat = pixel_values_videos.flatten(0, 1)
        video_position_ids_flat = video_position_ids.flatten(0, 1)
        
        video_features = self.get_video_features(
            pixel_values_videos_flat, video_position_ids_flat
        ).pooler_output
        
        # ... scatter like images ...
    
    # === STEP 5: Embed & scatter audio ===
    if input_features is not None and input_features_mask is not None:
        audio_output = self.get_audio_features(
            input_features, input_features_mask
        )
        audio_features = audio_output.pooler_output  # (batch, num_frames, 3840)
        audio_mask_from_encoder = audio_output.attention_mask  # True=valid
        
        # Strip padding tokens (false in mask)
        audio_features = audio_features[audio_mask_from_encoder.to(audio_features.device)]
        
        # ... scatter like images ...
    
    # === STEP 6: Build attention mask ===
    # Prepare position IDs
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values else 0
        position_ids = torch.arange(inputs_embeds.shape[1]) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)
    
    # Prepare causal/bidirectional masks
    if not isinstance(attention_mask, dict):  # dict means pre-prepared
        mask_kwargs = {
            "config": self.config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        
        # For bidirectional vision attention
        if self.config.get_text_config().use_bidirectional_attention == "vision":
            block_sequence_ids = torch.full(
                inputs_embeds.size()[:-1], -1, device=inputs_embeds.device
            )
            if mm_token_type_ids is not None:
                block_sequence_ids = get_block_sequence_ids_for_mask(
                    mm_token_type_ids, device=inputs_embeds.device
                )
            mask_kwargs["block_sequence_ids"] = block_sequence_ids
        
        # Create masks (causal or sliding)
        causal_mask_mapping = create_masks_for_generate(**mask_kwargs)
    
    # === STEP 7: Forward through language model ===
    outputs = self.language_model(
        attention_mask=causal_mask_mapping,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        return_dict=True,
        **kwargs,
    )
    
    # === STEP 8: Return ===
    return Gemma4UnifiedModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values else None,
        audio_hidden_states=audio_features if input_features else None,
        shared_kv_states=outputs.shared_kv_states,
    )
```

### Class: `Gemma4UnifiedForConditionalGeneration`

Top-level model for generation (inference + training):

```python
class Gemma4UnifiedForConditionalGeneration(Gemma4ForConditionalGeneration):
    def forward(
        self,
        input_ids: LongTensor | None = None,
        pixel_values: FloatTensor | None = None,
        pixel_values_videos: FloatTensor | None = None,
        input_features: FloatTensor | None = None,
        attention_mask: Tensor | None = None,
        input_features_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        image_position_ids: LongTensor | None = None,
        video_position_ids: LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: LongTensor | None = None,
        inputs_embeds: FloatTensor | None = None,
        labels: LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | Tensor = 0,
        **kwargs,
    ) -> Gemma4UnifiedCausalLMOutputWithPast:
        
        # Forward through model trunk
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        
        # Project hidden states → logits
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # Apply final logit softcapping if configured
        if (softcap := self.config.get_text_config().final_logit_softcapping) is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits, labels,
                self.config.get_text_config().vocab_size,
                **kwargs
            )
        
        return Gemma4UnifiedCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            audio_hidden_states=outputs.audio_hidden_states,
            shared_kv_states=outputs.shared_kv_states,
        )
```

---

## Input Preprocessing Requirements

### Text Input

**Type**: String or list of strings  
**Processor**: `GemmaTokenizer` (SentencePiece)

```python
tokenizer.encode(
    "What is in this image?",
    add_special_tokens=True,
)
# → [2, 1234, 310, ...] (token IDs)
```

**Special tokens**:
- `<bos>` (2): Start of sequence
- `<eos>` (1): End of sequence
- `<pad>` (0): Padding
- `<|image|>` (258880): Image placeholder
- `<|video|>` (258884): Video placeholder
- `<|audio|>` (258881): Audio placeholder

### Image Input

**Type**: PIL Image, numpy array (H, W, 3), or torch.Tensor (3, H, W)

**Processor**: `Gemma4UnifiedImageProcessor`

```python
processor.image_processor.preprocess(
    images=[pil_image_1, pil_image_2],
    do_resize=True,
    do_rescale=True,
    do_normalize=False,
    patch_size=16,
    max_soft_tokens=280,  # typical
    pooling_kernel_size=3,
    return_tensors="pt",
)
# Returns:
# {
#   "pixel_values": (batch, 280, 6912),
#   "image_position_ids": (batch, 280, 2),
#   "num_soft_tokens_per_image": [actual_count_1, actual_count_2],
# }
```

**Constraints**:
- Height and width must be divisible by `patch_size × pooling_kernel_size = 48` pixels after resize
- Images are resized via aspect-ratio-preserving algorithm
- No normalization applied (values in [0, 1] assumed after rescale)

### Video Input

**Type**: numpy array (num_frames, H, W, 3) or torch.Tensor

**Processor**: `Gemma4UnifiedVideoProcessor`

```python
processor.video_processor.preprocess(
    videos=[video_1, video_2],  # (num_frames, H, W, 3)
    do_resize=True,
    do_rescale=True,
    do_normalize=True,
    num_frames=32,  # sample frames
    patch_size=16,
    max_soft_tokens=70,  # typical (lower than images)
    pooling_kernel_size=3,
    return_tensors="pt",
)
# Returns:
# {
#   "pixel_values_videos": (batch, num_frames, 70, 6912),
#   "video_position_ids": (batch, num_frames, 70, 2),
#   "num_soft_tokens_per_video": [count_1, count_2],
# }
```

### Audio Input

**Type**: numpy array (num_samples,), list of floats, or torch.Tensor

**Processor**: `Gemma4UnifiedAudioFeatureExtractor`

```python
processor.feature_extractor(
    raw_speech=[audio_1, audio_2],  # 16 kHz, 1D
    sampling_rate=16_000,
    padding="longest",
    max_length=None,  # optional max frames
    truncation=True,
    return_tensors="pt",
)
# Returns:
# {
#   "input_features": (batch, max_frames, 640),
#   "input_features_mask": (batch, max_frames),
# }
```

**Constraints**:
- Sampling rate: **16 kHz** (hardcoded)
- Audio frame size: **640 samples** = 40 ms
- No mel-spectrogram; raw waveform used directly

### Unified Processing: `Gemma4UnifiedProcessor`

Coordinates all modalities + text:

```python
processor = Gemma4UnifiedProcessor.from_pretrained("google/gemma-4-12B-it")

inputs = processor(
    images=[pil_image],
    videos=None,
    audio=None,
    text="Describe this image.",
    padding=True,
    return_tensors="pt",
)
# Returns dict with:
# - input_ids: (batch, seq_len)
# - attention_mask: (batch, seq_len)
# - pixel_values: (batch, max_soft_tokens, 6912)
# - image_position_ids: (batch, max_soft_tokens, 2)
# - num_soft_tokens_per_image: list[int]
# - mm_token_type_ids: (batch, seq_len) [for bidirectional vision attention]
```

**Processor responsibilities**:
1. Parse text & generate image/audio placeholders
2. Process images/videos/audio separately
3. Merge results into unified input dict
4. Compute token budgets for dynamic prompt construction

---

## Special Features & Optimizations

### 1. Hybrid Attention Schedule: 5:1 Sliding + Full

**Why**: Balance context coverage with efficiency.

- **5 sliding-window layers** (1024-token window): O(n) complexity, fast
- **1 full-attention layer** (all history): Global context, expressive

**Pattern** (30-layer variant):
```
Layer 0-4: sliding (window=1024)
Layer 5:   full
Layer 6-10: sliding
Layer 11:  full
...
Layer 26-30: sliding
Layer 31:  full (last layer, forced)
```

**RoPE per layer type**:
- Sliding: θ = 10,000 (default), rope_type = "default"
- Full: θ = 1,000,000 (1M), rope_type = "proportional", partial_rotary_factor = 0.25

**Rationale**: Full attention needs larger θ to model long-range dependencies without interpolation.

### 2. Bidirectional Vision Attention

**Why**: Vision tokens should see all text; text shouldn't "cheat" by seeing future text.

**Mechanism**:
- Token block IDs are passed: `block_sequence_ids`
  - Vision tokens (image/video): block ID = same value (tokens within same image attend bidirectionally)
  - Text tokens: block ID = unique per token (causality preserved)
- `create_masks_for_generate(..., block_sequence_ids=...)` generates per-type attention masks

**Example**:
```
Prompt: "Describe <image>. Q: What's the color?"

Token block IDs:
[image_token_1, image_token_2, ..., image_token_280] → block_id = 0 (all same)
["Describe", "<|image|>", ".", "Q", ":", ...] → block_ids = 1, 1, 2, 3, 4, ...

Attention:
- All 280 image tokens attend to each other (bidirectional)
- Text tokens attend only backwards (causal)
```

### 3. KV Sharing (Optional, Not Default)

**Why**: Reduce parameters & memory for very large models.

**Mechanism**:
- Last N layers **don't have** K, V projections
- Instead, they reuse K, V from earlier layers (e.g., layer 42 reuses layer 37)
- `shared_kv_states` dict passes K, V between layers

**Example** (num_kv_shared_layers=4, 48 total layers):
```
first_kv_shared_layer_idx = 48 - 4 = 44

Layers 0-43: have k_proj, v_proj (independent)
Layers 44-47: reuse K, V from layers 40, 41, 42, 43
```

### 4. Per-Layer Learnable Scalar

**Why**: Stabilize training, allow per-layer output scaling.

**Mechanism**:
```python
self.register_buffer("layer_scalar", torch.ones(1), persistent=True)  # trainable
hidden_states *= self.layer_scalar  # scale before residual
```

Initialized to 1; learned to optimize loss. Acts like layer-wise adaptive learning rates.

### 5. Double-Wide MLP (Optional)

**Why**: Increase model capacity while maintaining efficiency.

**Feature**: `use_double_wide_mlp: bool`

When True:
- FFN gate and up projections are **fused**: `(gate_proj(x) ⊙ act(up_proj(x)))`
- Single large linear layer vs. two separate ones

Not the default; requires specific training recipes.

### 6. Alternative Attention: K=V Sharing

**Feature**: `attention_k_eq_v: bool`

When True (typically in full-attention layers with global heads):
- K and V projections share weights
- Q, K, V are derived from the same projection matrix → compresses model
- Used with `num_global_key_value_heads=1` for extreme compression

Example (Gemma-4-12B-assistant):
```
text_config=Gemma4UnifiedTextConfig(
    num_attention_heads=16,
    num_key_value_heads=8,
    num_global_key_value_heads=1,  # only 1 global KV head
    attention_k_eq_v=True,          # K=V
    ...
)
```

---

## Implementation Guide for Other Languages

### System Architecture Overview

```
Input Layer (Text + Modalities)
    ↓
Tokenization & Preprocessing
    ├─ Text: SentencePiece tokenizer → token IDs
    ├─ Image: Patchify (16px) → Merge (3×3) → position tracking
    ├─ Video: Per-frame patchification
    ├─ Audio: Chunk frames (640 samples)
    ↓
Embeddings & Projection
    ├─ Text: Embedding layer + RoPE
    ├─ Vision: Dense(patch) + factorized 2D posemb + multimodal_proj
    ├─ Audio: Direct multimodal projection
    ↓
Merged Embedding Space (unified 3840-dim vectors)
    ↓
Hybrid Transformer Blocks (30 or 48 layers)
    ├─ Per-block: Determine layer_type (sliding vs. full)
    ├─ Per-block: Load layer-specific RoPE params
    ├─ Attention: Sliding window (1024) or Full (262k)
    ├─ Attention: Bidirectional (vision) or causal (text)
    ├─ FFN: Gated linear unit (GLU) style
    ├─ Residual + RMSNorm + per-layer scalar
    ↓
Final RMSNorm
    ↓
LM Head (Linear: 3840 → 262144 vocab)
    ↓
Logits → Loss / Generation
```

### Key Implementation Details

#### 1. **Modular Config System**

```
Config Hierarchy:
├─ Gemma4UnifiedConfig (top-level)
│  ├─ text_config: Gemma4UnifiedTextConfig
│  ├─ vision_config: Gemma4UnifiedVisionConfig
│  ├─ audio_config: Gemma4UnifiedAudioConfig
│  └─ modality_token_ids (boi, eoi, boa, eoa, etc.)
```

**In other languages**: Load from JSON, validate schema, instantiate typed objects.

#### 2. **Hybrid Attention Scheduling**

```
For layer_idx in 0..num_layers:
    layer_type = config.layer_types[layer_idx]
    
    if layer_type == "sliding_attention":
        window_size = config.sliding_window  # 1024
        rope_params = config.rope_parameters["sliding_attention"]
        # window_size_in_tokens = 1024
        
    elif layer_type == "full_attention":
        window_size = FULL  # no limit
        rope_params = config.rope_parameters["full_attention"]
        # includes global_head_dim for larger head
```

**Implementation tip**: Pre-compute layer type array and RoPE init at startup, cache.

#### 3. **Per-Layer RoPE Computation**

```
For each layer_type in {"sliding_attention", "full_attention"}:
    rope_theta = rope_parameters[layer_type]["rope_theta"]
    
    if rope_type == "default":
        dim = head_dim
        inv_freq = 1 / (rope_theta ** (arange(0, dim, 2) / dim))
    
    elif rope_type == "proportional":
        head_dim_key = rope_parameters[layer_type].get("head_dim_key")
        if head_dim_key == "global_head_dim":
            dim = config.global_head_dim  # 512, not 256
        else:
            dim = config.head_dim
        inv_freq = ...  # same formula
```

**Implementation tip**: Store both `sliding_inv_freq` and `full_inv_freq` buffers; index by layer_type at runtime.

#### 4. **Patches Merge Algorithm**

This is the most complex, most novel part. **Pseudo-code**:

```
function patches_merge(patches, positions_xy, target_length):
    # patches: (L, 768)  — teacher patches
    # positions_xy: (L, 2)  — (x, y) coordinates
    # target_length: (L / 9) for 3×3 merge
    
    patch_size = isqrt(patches.shape[-1] / 3)  # = 16
    k = isqrt(patches.shape[-2] / target_length)  # = 3
    
    # Step 1: Compute kernel group indices
    kernel_idxs = floor(positions_xy / k)
    
    # Step 2: Compute spatial ordering
    max_x = max(positions_xy[:, 0]) + 1
    num_patches_from_top_left = (
        k * k * kernel_idxs[:, 0] +
        k * max_x * kernel_idxs[:, 1]
    )
    position_within_kernel = mod(positions_xy, k)
    num_patches_from_top_left_of_kernel = (
        position_within_kernel[:, 0] +
        position_within_kernel[:, 1] * k
    )
    target_ordering = (
        num_patches_from_top_left_of_kernel +
        num_patches_from_top_left
    )
    
    # Step 3: Argsort to get permutation
    perm = argsort(target_ordering)
    kernel_ordered_patches = patches[perm]
    kernel_ordered_positions = positions_xy[perm]
    
    # Step 4: Reshape into k×k blocks
    # (L, 768) → (target_length, 9, 16, 16, 3)
    kernel_ordered_patches = reshape(
        kernel_ordered_patches,
        (target_length, k*k, patch_size, patch_size, 3)
    )
    
    # Step 5: Rearrange spatial dimensions
    # (target_length, 9, 16, 16, 3)
    # → (target_length, 3, 3, 16, 16, 3)
    kernel_ordered_patches = reshape(
        kernel_ordered_patches,
        (target_length, k, k, patch_size, patch_size, 3)
    )
    
    # Permute (l, k, p, k, q, c) → (l, k, p, k, q, c)
    kernel_ordered_patches = permute(
        kernel_ordered_patches,
        (0, 1, 3, 2, 4, 5)  # spatial reorder
    )
    
    # (target_length, 48, 48, 3)
    merged_patches = reshape(
        kernel_ordered_patches,
        (target_length, k*patch_size, k*patch_size, 3)
    )  # → (target_length, 6912)
    
    # Step 6: Update positions
    kernel_ordered_positions = reshape(
        kernel_ordered_positions,
        (target_length, k*k, 2)
    )
    new_positions = floor(
        min(kernel_ordered_positions, axis=1) / k
    )
    
    return merged_patches, new_positions
```

**Key implementation tips**:
- Use tile/block operations efficiently in your language
- Handle dimension ordering carefully (different from PyTorch if needed)
- Test against ground truth (compare outputs with PyTorch reference)

#### 5. **Vision Embedder Forward Pass**

```
def vision_embedder_forward(pixel_values, image_position_ids):
    # pixel_values: (batch, max_patches, 6912)
    # image_position_ids: (batch, max_patches, 2)
    
    # Step 1: Patch embedding
    x = layer_norm(pixel_values)
    x = linear_dense(x)  # (batch, max_patches, 3840)
    x = layer_norm(x)
    
    # Step 2: Factorized 2D positional embeddings
    pos_embedding: (1120, 2, 3840)  # lookup table
    
    # For each sample, each patch:
    clamped_pos = clamp(image_position_ids, min=0)
    valid_mask = (image_position_ids != -1)  # (batch, max_patches, 2, 1)
    
    pos_embs = pos_embedding[clamped_pos[:, :, 0], :]
    # → (batch, max_patches, 2, 3840)
    
    pos_embs = pos_embs * valid_mask  # zero out invalid
    pos_embs = sum(pos_embs, axis=2)  # (batch, max_patches, 3840)
    
    x = x + pos_embs
    x = layer_norm(x)
    
    # Step 3: Multimodal embedder (RMSNorm → Linear)
    x = rms_norm(x)
    x = linear_proj(x)  # project to text hidden_size (3840)
    
    # Step 4: Strip padding
    padding_mask = all(image_position_ids == -1, axis=-1)  # (batch, max_patches)
    x = x[~padding_mask]  # flatten & filter
    
    return x  # (total_valid_patches, 3840)
```

#### 6. **Attention Mask Construction for Bidirectional Vision**

```
def create_attention_mask(
    seq_len,
    block_sequence_ids,  # None or (batch, seq_len)
    use_bidirectional_attention,
):
    """
    Returns attention mask of shape (batch, 1, seq_len, seq_len) or similar.
    1 = attend, -inf = mask out
    """
    
    if block_sequence_ids is None:
        # Standard causal mask
        return causal_mask(seq_len)  # upper triangular -inf
    
    # Bidirectional vision attention
    if use_bidirectional_attention == "vision":
        # Create mask based on block IDs
        # Same block ID → attend both directions
        # Different block IDs → causal (earlier attends to later only if causal)
        
        mask = zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                if block_sequence_ids[i] == block_sequence_ids[j]:
                    # Same block: bidirectional
                    mask[i, j] = 0
                elif j > i:
                    # Different block, j > i: mask out (future)
                    mask[i, j] = -inf
                else:
                    # Different block, j <= i: attend (past)
                    mask[i, j] = 0
        
        return mask
```

#### 7. **Tokenizer Integration**

Gemma4 uses **SentencePiece** (Google's subword tokenizer):

```
# Load from .model file
tokenizer = SentencePieceTokenizer.load("gemma4.model")

# Encode text
token_ids = tokenizer.encode("Hello world")  # [2, 1234, 5678, 1]

# Add special tokens
token_ids = tokenizer.encode(
    "Describe this image.",
    add_bos=True,
    add_eos=True
)

# Decode
text = tokenizer.decode([2, 1234, 5678, 1])
```

**Special tokens**:
- `<image>` (258880) - vision placeholder
- `<audio>` (258881) - audio placeholder
- `<video>` (258884) - video placeholder
- Begin/end: `<|image|>` (255999), `<|end_image|>` (258882), etc.

#### 8. **Weight Loading from Safetensors**

Gemma4 Unified weights are stored in safetensors format:

```
model.safetensors  — all parameters
tokenizer.model    — SentencePiece model
config.json        — configuration
```

Key weight names:
```
— Embeddings
model.embed_tokens.weight: (262144, 3840)
model.rotary_emb.{layer_type}_inv_freq: per-layer

— Transformer blocks
model.layers.{idx}.self_attn.{q,k,v}_proj.weight: (*, 3840)
model.layers.{idx}.self_attn.{q,k,v}_norm.weight: (256 or 512,)
model.layers.{idx}.mlp.gate_proj.weight: (9216, 3840)
model.layers.{idx}.mlp.up_proj.weight: (9216, 3840)
model.layers.{idx}.mlp.down_proj.weight: (3840, 9216)
model.layers.{idx}.input_layernorm.weight: (3840,)
model.layers.{idx}.post_attention_layernorm.weight: (3840,)
model.layers.{idx}.pre_feedforward_layernorm.weight: (3840,)
model.layers.{idx}.post_feedforward_layernorm.weight: (3840,)
model.layers.{idx}.layer_scalar: (1,)  — scalar multiplier

— Vision embedder
embed_vision.patch_ln1.weight, bias: (6912,)
embed_vision.patch_dense.weight, bias: (3840, 6912)
embed_vision.patch_ln2.weight, bias: (3840,)
embed_vision.pos_embedding: (1120, 2, 3840)
embed_vision.pos_norm.weight, bias: (3840,)
embed_vision.multimodal_embedder.embedding_pre_projection_norm.weight: (3840,)
embed_vision.multimodal_embedder.embedding_projection.weight, bias: (3840, 3840)

— Audio embedder (similar structure)
embed_audio.embedding_pre_projection_norm.weight: (640,)
embed_audio.embedding_projection.weight, bias: (3840, 640)

— LM head
lm_head.weight: (262144, 3840)
```

#### 9. **Generation (Inference)**

```
def generate(
    model,
    prompt_tokens,  # (batch, seq_len)
    pixel_values,  # optional
    input_features,  # optional
    max_new_tokens=100,
    temperature=0.7,
    top_k=40,
    top_p=0.95,
):
    """Autoregressive generation with KV cache."""
    
    past_key_values = None
    
    for step in range(max_new_tokens):
        # Prepare inputs
        if step == 0:
            # First iteration: full prompt
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=prompt_tokens,
                pixel_values=pixel_values,
                input_features=input_features,
                is_first_iteration=True,
            )
        else:
            # Subsequent iterations: only last token
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=prompt_tokens[:, -1:],  # only last token
                past_key_values=past_key_values,
                is_first_iteration=False,
            )
        
        # Forward pass
        outputs = model(**model_inputs, use_cache=True)
        logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
        past_key_values = outputs.past_key_values
        
        # Sample next token
        probs = softmax(logits / temperature, axis=-1)
        # ... apply top_k, top_p filtering ...
        next_token = sample_from(probs)  # (batch, 1)
        
        # Append to sequence
        prompt_tokens = concat([prompt_tokens, next_token], axis=1)
        
        # Check stopping condition (EOS token)
        if all(next_token == eos_token_id):
            break
    
    return prompt_tokens
```

---

## Summary & Key Takeaways

| Feature | Detail |
|---------|--------|
| **Architecture** | Decoder-only, encoder-free, omni-modal (text + image + video + audio) |
| **Context** | 262,144 tokens (covers multiple images, long videos, extended audio) |
| **Vision** | 16px teacher patches → 3×3 merge → 48px model patches (6912 dims) → dense projection → factorized 2D posemb |
| **Audio** | Raw 16kHz waveform → 640-sample frames → direct projection (no mel, no FFT) |
| **Attention** | Hybrid: 5 sliding (1024 tokens) + 1 full-context per repeat, bidirectional for vision |
| **RoPE** | Per-layer-type: sliding θ=10k, full θ=1M with partial rotary (0.25) |
| **Optimizations** | KV sharing, per-layer scalar, learnable MLP variants, K=V projection sharing |
| **Key Innovation** | **Encoder-free**; no separate towers; all modalities projected into shared text space |

**For implementation in other languages**:
1. Load configuration from JSON (nested sub-configs)
2. Implement per-layer-type RoPE and attention scheduling
3. Implement `patches_merge` correctly (most complex part)
4. Thread `shared_kv_states` through layers if using KV sharing
5. Handle bidirectional attention via block_sequence_ids masking
6. Load weights from safetensors; map names carefully
7. Test against HF reference outputs on small examples first

---

**Document Version**: 1.0 (2026)  
**Last Updated**: June 5, 2026  
**Source**: HuggingFace Transformers repository, Gemma4Unified model code  
**Reference Implementation**: PyTorch in `src/transformers/models/gemma4_unified/`
