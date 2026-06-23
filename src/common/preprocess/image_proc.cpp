#define STB_IMAGE_IMPLEMENTATION
#include "image_proc.hpp"
#include <stb/stb_image.h>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cstring>

// ---------------------------------------------------------------------------
// Bilinear resize (H,W,C) → (target_H, target_W, C) in float
// ---------------------------------------------------------------------------
static std::vector<float> resize_bilinear(
    const float* src, int src_h, int src_w,
    int dst_h, int dst_w, int C
) {
    std::vector<float> dst(dst_h * dst_w * C);
    float yscale = float(src_h) / float(dst_h);
    float xscale = float(src_w) / float(dst_w);

    for (int y = 0; y < dst_h; ++y) {
        float fy = (y + 0.5f) * yscale - 0.5f;
        int   y0 = std::max(0, (int)std::floor(fy));
        int   y1 = std::min(src_h - 1, y0 + 1);
        float dy = fy - y0;

        for (int x = 0; x < dst_w; ++x) {
            float fx = (x + 0.5f) * xscale - 0.5f;
            int   x0 = std::max(0, (int)std::floor(fx));
            int   x1 = std::min(src_w - 1, x0 + 1);
            float dx = fx - x0;

            for (int c = 0; c < C; ++c) {
                float v00 = src[(y0 * src_w + x0) * C + c];
                float v01 = src[(y0 * src_w + x1) * C + c];
                float v10 = src[(y1 * src_w + x0) * C + c];
                float v11 = src[(y1 * src_w + x1) * C + c];
                dst[(y * dst_w + x) * C + c] =
                    v00 * (1-dy) * (1-dx) + v01 * (1-dy) * dx +
                    v10 * dy    * (1-dx) + v11 * dy    * dx;
            }
        }
    }
    return dst;
}

// ---------------------------------------------------------------------------
// Patchify: (H,W,C) pixels → (num_patches, ps*ps*C) + (num_patches, 2) positions
// Input pixels are float in [0, 1] (pre-scaled).
// ---------------------------------------------------------------------------
static void patchify(
    const float* pixels, int H, int W, int C, int ps,
    std::vector<float>& patches,
    std::vector<std::array<int,2>>& positions
) {
    int ph = H / ps, pw = W / ps;
    int patch_dim = ps * ps * C;
    patches.resize(ph * pw * patch_dim);
    positions.resize(ph * pw);

    int idx = 0;
    for (int py = 0; py < ph; ++py) {
        for (int px = 0; px < pw; ++px) {
            // Extract patch (px, py) into row-major (ps*ps*C) — channel-last
            float* dst = patches.data() + idx * patch_dim;
            for (int dy = 0; dy < ps; ++dy)
                for (int dx = 0; dx < ps; ++dx)
                    for (int c = 0; c < C; ++c)
                        dst[(dy * ps + dx) * C + c] =
                            pixels[((py * ps + dy) * W + (px * ps + dx)) * C + c];
            positions[idx] = {px, py};
            ++idx;
        }
    }
}

// ---------------------------------------------------------------------------
// patches_merge: 3×3 spatial pooling of teacher patches → model patches
// teacher_patches: (L, 768)  pos_xy: (L, 2)
// target_length: L / (k*k)
// ---------------------------------------------------------------------------
struct MergedPatches {
    std::vector<float>             data;      // (target_length, 6912)
    std::vector<std::array<int,2>> positions; // (target_length, 2) — divided by k
};

static MergedPatches patches_merge(
    const std::vector<float>& teacher,
    const std::vector<std::array<int,2>>& pos_xy,
    int target_length,
    int k = 3
) {
    int L = (int)pos_xy.size();
    int patch_dim = (int)(teacher.size() / L);

    int max_x = 0;
    for (auto& p : pos_xy) max_x = std::max(max_x, p[0]);
    max_x += 1;

    std::vector<int> target_order(L);
    for (int i = 0; i < L; ++i) {
        int x = pos_xy[i][0], y = pos_xy[i][1];
        int kx = x / k, ky = y / k;
        int within_x = x % k, within_y = y % k;
        int num_from_tl = k * k * kx + k * max_x * ky;
        int num_in_kernel = within_x + within_y * k;
        target_order[i] = num_in_kernel + num_from_tl;
    }

    // argsort
    std::vector<int> perm(L);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        return target_order[a] < target_order[b];
    });

    // ordered patches: (target_length, k*k, patch_dim)
    // then reshape to (target_length, k*patch_size, k*patch_size, C)
    // We achieve the permute (0,1,3,2,4,5) by careful index arithmetic.
    // patch_dim = ps*ps*C where ps=16, C=3 → patch is (ps, ps, C)
    int ps = 16, C = 3;
    int merged_dim = k * k * patch_dim; // 9 * 768 = 6912

    MergedPatches mp;
    mp.data.resize(target_length * merged_dim);
    mp.positions.resize(target_length);

    for (int m = 0; m < target_length; ++m) {
        // 9 teacher patches for this model patch
        // perm[m*9 + j] gives the j-th teacher patch index in the 3×3 block.
        // The within-kernel order is: j = within_x + within_y*k
        // We want to assemble them into a (3,16, 3,16, 3) tensor and permute
        // axes (0,1,3,2,4,5) → (3,16, 3,16, 3) → (48, 48, 3) = (6912,).
        float* dst = mp.data.data() + m * merged_dim;

        // Accumulate: for each pixel in the merged 48×48 image:
        // merged_y in [0,48), merged_x in [0,48)
        // block_y = merged_y / ps → ky in [0,k); local_y = merged_y % ps
        // block_x = merged_x / ps → kx in [0,k); local_x = merged_x % ps
        // kernel slot j = kx + ky*k (within-kernel index)
        // teacher patch index = perm[m*9 + j]
        // value at (local_y, local_x, c)
        for (int ky = 0; ky < k; ++ky) {
            for (int kx = 0; kx < k; ++kx) {
                int j = kx + ky * k;
                int teacher_idx = perm[m * (k * k) + j];
                const float* src = teacher.data() + teacher_idx * patch_dim;

                for (int ly = 0; ly < ps; ++ly) {
                    for (int lx = 0; lx < ps; ++lx) {
                        for (int c = 0; c < C; ++c) {
                            int dst_y = ky * ps + ly;
                            int dst_x = kx * ps + lx;
                            dst[(dst_y * (k * ps) + dst_x) * C + c] =
                                src[(ly * ps + lx) * C + c];
                        }
                    }
                }
            }
        }

        // New position = min(positions of the 9 patches) / k
        int min_px = INT_MAX, min_py = INT_MAX;
        for (int j = 0; j < k * k; ++j) {
            int ti = perm[m * (k * k) + j];
            min_px = std::min(min_px, pos_xy[ti][0]);
            min_py = std::min(min_py, pos_xy[ti][1]);
        }
        mp.positions[m] = {min_px / k, min_py / k};
    }

    return mp;
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------
ImageInput preprocess_image(
    const std::string& path,
    int patch_size, int pooling_kernel, int max_soft_tokens,
    float rescale_factor, bool do_rescale
) {
    int W, H, C_img;
    uint8_t* raw = stbi_load(path.c_str(), &W, &H, &C_img, 3);
    if (!raw) throw std::runtime_error("stbi_load failed: " + path);

    // Float conversion with processor-configured rescale behavior.
    std::vector<float> pixels(H * W * 3);
    for (int i = 0; i < H * W * 3; ++i)
        pixels[i] = do_rescale ? raw[i] * rescale_factor : (float)raw[i];
    stbi_image_free(raw);

    // Compute target size
    int model_patch_size = patch_size * pooling_kernel; // 48
    int max_teacher_patches = max_soft_tokens * pooling_kernel * pooling_kernel; // 2520
    float target_px = float(max_teacher_patches) * float(patch_size * patch_size);
    float factor = std::sqrt(target_px / float(H * W));

    int th = std::max(model_patch_size, (int)(H * factor / model_patch_size) * model_patch_size);
    int tw = std::max(model_patch_size, (int)(W * factor / model_patch_size) * model_patch_size);

    // Resize
    auto resized = resize_bilinear(pixels.data(), H, W, th, tw, 3);

    // Patchify into 16×16 teacher patches
    std::vector<float> teacher;
    std::vector<std::array<int,2>> pos_xy;
    patchify(resized.data(), th, tw, 3, patch_size, teacher, pos_xy);

    int L = (int)pos_xy.size();
    int target_length = L / (pooling_kernel * pooling_kernel);

    // patches_merge
    auto mp = patches_merge(teacher, pos_xy, target_length, pooling_kernel);

    // Pad to max_soft_tokens
    int merged_dim = pooling_kernel * patch_size * pooling_kernel * patch_size * 3; // 6912
    ImageInput out;
    out.num_valid_patches = std::min(target_length, max_soft_tokens);
    out.max_patches = max_soft_tokens;
    out.pixel_values.assign(max_soft_tokens * merged_dim, 0.0f);
    out.position_ids.assign(max_soft_tokens, {-1, -1});

    for (int i = 0; i < out.num_valid_patches; ++i) {
        std::memcpy(out.pixel_values.data() + i * merged_dim,
                    mp.data.data() + i * merged_dim,
                    merged_dim * sizeof(float));
        out.position_ids[i] = mp.positions[i];
    }

    return out;
}
