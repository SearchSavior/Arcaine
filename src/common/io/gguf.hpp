#pragma once
#include "tensor_view.hpp"
#include <mio/mio.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct GgufMeta {
    enum class Kind { None, Int, Float, Bool, Str, ArrInt, ArrFloat, ArrBool, ArrStr };
    Kind kind = Kind::None;
    int64_t i = 0;  double f = 0;  bool b = false;  std::string s;
    std::vector<int64_t> ai;  std::vector<double> af;
    std::vector<uint8_t> ab;  std::vector<std::string> as;
};

class GgufFile {
public:
    explicit GgufFile(const std::string& path) {
        std::error_code ec;
        mmap_.map(path, ec);
        if (ec) throw std::runtime_error("mmap failed on " + path + ": " + ec.message());
        base_ = mmap_.data();
        ptr_ = base_;
        end_ = base_ + mmap_.size();

        if (read_u32() != 0x46554747u)
            throw std::runtime_error("GGUF: bad magic in " + path);
        uint32_t version = read_u32();
        if (version < 2 || version > 3)
            throw std::runtime_error("GGUF: unsupported version " + std::to_string(version));

        uint64_t tensor_count = read_u64();
        uint64_t kv_count = read_u64();
        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key = read_string();
            uint32_t type = read_u32();
            metadata_[key] = read_metadata_value(type);
        }

        struct PendingTensor {
            std::string name;
            std::vector<int64_t> shape;
            std::string dtype;
            uint64_t offset;
            size_t nbytes;
        };
        std::vector<PendingTensor> pending;
        pending.reserve((size_t)tensor_count);
        for (uint64_t i = 0; i < tensor_count; ++i) {
            PendingTensor t;
            t.name = read_string();
            uint32_t n_dims = read_u32();
            std::vector<uint64_t> dims(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) dims[d] = read_u64();
            uint32_t ggml_type = read_u32();
            t.offset = read_u64();

            t.shape.reserve(n_dims);
            for (auto it = dims.rbegin(); it != dims.rend(); ++it)
                t.shape.push_back((int64_t)*it);
            t.dtype = dtype_name(ggml_type);
            t.nbytes = tensor_nbytes(dims, ggml_type);
            pending.push_back(std::move(t));
        }

        uint64_t alignment = 32;
        uint32_t align_val = 0;
        if (get_u32("general.alignment", align_val) && align_val != 0) alignment = align_val;
        const char* data_start = base_ + align_up((size_t)(ptr_ - base_), (size_t)alignment);
        if (data_start > end_) throw std::runtime_error("GGUF: tensor data starts past EOF");

        for (const PendingTensor& p : pending) {
            const char* data = data_start + p.offset;
            if (data < base_ || data + p.nbytes > end_)
                throw std::runtime_error("GGUF: tensor data exceeds file: " + p.name);
            TensorView tv;
            tv.dtype = p.dtype;
            tv.shape = p.shape;
            tv.data = data;
            tv.nbytes = p.nbytes;
            tensors_[p.name] = std::move(tv);
        }
    }

    const TensorView& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("GGUF: tensor not found: " + name);
        return it->second;
    }

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    size_t num_tensors() const { return tensors_.size(); }

    // Typed metadata accessors (non-throwing; return false on absence/wrong-kind)
    bool get_u32 (const std::string& k, uint32_t& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::Int) return false;
        out = (uint32_t)it->second.i; return true;
    }
    bool get_i32 (const std::string& k, int32_t& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::Int) return false;
        out = (int32_t)it->second.i; return true;
    }
    bool get_f32 (const std::string& k, float& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::Float) return false;
        out = (float)it->second.f; return true;
    }
    bool get_bool(const std::string& k, bool& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::Bool) return false;
        out = it->second.b; return true;
    }
    bool get_str (const std::string& k, std::string& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::Str) return false;
        out = it->second.s; return true;
    }
    bool get_i32_array (const std::string& k, std::vector<int32_t>& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::ArrInt) return false;
        out.assign(it->second.ai.begin(), it->second.ai.end()); return true;
    }
    bool get_f32_array (const std::string& k, std::vector<float>& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::ArrFloat) return false;
        out.assign(it->second.af.begin(), it->second.af.end()); return true;
    }
    bool get_bool_array(const std::string& k, std::vector<uint8_t>& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::ArrBool) return false;
        out = it->second.ab; return true;
    }
    bool get_str_array (const std::string& k, std::vector<std::string>& out) const {
        auto it = metadata_.find(k);
        if (it == metadata_.end() || it->second.kind != GgufMeta::Kind::ArrStr) return false;
        out = it->second.as; return true;
    }

private:
    uint32_t read_u32() {
        require(4);
        uint32_t v;
        std::memcpy(&v, ptr_, 4);
        ptr_ += 4;
        return v;
    }

    uint64_t read_u64() {
        require(8);
        uint64_t v;
        std::memcpy(&v, ptr_, 8);
        ptr_ += 8;
        return v;
    }

    std::string read_string() {
        uint64_t n = read_u64();
        require((size_t)n);
        std::string s(ptr_, ptr_ + n);
        ptr_ += n;
        return s;
    }

    void require(size_t n) const {
        if ((size_t)(end_ - ptr_) < n) throw std::runtime_error("GGUF: truncated file");
    }

    GgufMeta read_metadata_value(uint32_t type) {
        enum : uint32_t {
            U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5,
            F32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, U64 = 10,
            I64 = 11, F64 = 12
        };
        GgufMeta m;
        switch (type) {
            case U8:  { require(1); uint8_t v;  std::memcpy(&v, ptr_, 1); ptr_ += 1; m.kind = GgufMeta::Kind::Int; m.i = v; return m; }
            case I8:  { require(1); int8_t v;   std::memcpy(&v, ptr_, 1); ptr_ += 1; m.kind = GgufMeta::Kind::Int; m.i = v; return m; }
            case U16: { require(2); uint16_t v; std::memcpy(&v, ptr_, 2); ptr_ += 2; m.kind = GgufMeta::Kind::Int; m.i = v; return m; }
            case I16: { require(2); int16_t v;  std::memcpy(&v, ptr_, 2); ptr_ += 2; m.kind = GgufMeta::Kind::Int; m.i = v; return m; }
            case U32: { m.kind = GgufMeta::Kind::Int; m.i = read_u32(); return m; }
            case I32: { require(4); int32_t v;  std::memcpy(&v, ptr_, 4); ptr_ += 4; m.kind = GgufMeta::Kind::Int; m.i = v; return m; }
            case F32: { require(4); float v;    std::memcpy(&v, ptr_, 4); ptr_ += 4; m.kind = GgufMeta::Kind::Float; m.f = v; return m; }
            case BOOL:{ require(1); uint8_t v;  std::memcpy(&v, ptr_, 1); ptr_ += 1; m.kind = GgufMeta::Kind::Bool; m.b = v != 0; return m; }
            case STRING: { m.kind = GgufMeta::Kind::Str; m.s = read_string(); return m; }
            case ARRAY: {
                uint32_t elem = read_u32();
                uint64_t n = read_u64();
                if (elem == STRING) {
                    m.kind = GgufMeta::Kind::ArrStr;
                    m.as.reserve((size_t)n);
                    for (uint64_t i = 0; i < n; ++i) m.as.push_back(read_string());
                } else if (elem == F32 || elem == F64) {
                    m.kind = GgufMeta::Kind::ArrFloat;
                    m.af.reserve((size_t)n);
                    for (uint64_t i = 0; i < n; ++i) m.af.push_back(read_metadata_value(elem).f);
                } else if (elem == BOOL) {
                    m.kind = GgufMeta::Kind::ArrBool;
                    m.ab.reserve((size_t)n);
                    for (uint64_t i = 0; i < n; ++i) m.ab.push_back(read_metadata_value(elem).b ? 1 : 0);
                } else {
                    m.kind = GgufMeta::Kind::ArrInt;
                    m.ai.reserve((size_t)n);
                    for (uint64_t i = 0; i < n; ++i) m.ai.push_back(read_metadata_value(elem).i);
                }
                return m;
            }
            case U64: { m.kind = GgufMeta::Kind::Int; m.i = (int64_t)read_u64(); return m; }
            case I64: { require(8); int64_t v;  std::memcpy(&v, ptr_, 8); ptr_ += 8; m.kind = GgufMeta::Kind::Int; m.i = v; return m; }
            case F64: { require(8); double v;   std::memcpy(&v, ptr_, 8); ptr_ += 8; m.kind = GgufMeta::Kind::Float; m.f = v; return m; }
            default: throw std::runtime_error("GGUF: unknown metadata type " + std::to_string(type));
        }
    }

    static size_t align_up(size_t v, size_t a) {
        return ((v + a - 1) / a) * a;
    }

    static std::string dtype_name(uint32_t ggml_type) {
        switch (ggml_type) {
            case 0: return "F32";
            case 1: return "F16";
            case 8: return "Q8_0";
            case 30: return "BF16";
            default:
                throw std::runtime_error("GGUF: unsupported tensor type " + std::to_string(ggml_type));
        }
    }

    static size_t tensor_nbytes(const std::vector<uint64_t>& dims, uint32_t ggml_type) {
        size_t numel = 1;
        for (uint64_t d : dims) numel *= (size_t)d;
        switch (ggml_type) {
            case 0: return numel * 4;
            case 1: return numel * 2;
            case 30: return numel * 2;
            case 8: {
                if (dims.empty() || dims[0] % 32 != 0)
                    throw std::runtime_error("GGUF: Q8_0 tensor first dimension must be divisible by 32");
                return numel / 32 * (2 + 32);
            }
            default: throw std::runtime_error("GGUF: unsupported tensor type " + std::to_string(ggml_type));
        }
    }

    mio::mmap_source mmap_;
    const char* base_ = nullptr;
    const char* ptr_ = nullptr;
    const char* end_ = nullptr;
    std::unordered_map<std::string, TensorView> tensors_;
    std::unordered_map<std::string, GgufMeta> metadata_;
};
