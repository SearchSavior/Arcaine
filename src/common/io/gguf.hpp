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
            metadata_u64_[key] = read_metadata_value(type);
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
        auto it = metadata_u64_.find("general.alignment");
        if (it != metadata_u64_.end() && it->second != 0) alignment = it->second;
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

    uint64_t read_metadata_value(uint32_t type) {
        enum : uint32_t {
            U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5,
            F32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, U64 = 10,
            I64 = 11, F64 = 12
        };
        switch (type) {
            case U8: require(1); ptr_ += 1; return 0;
            case I8: require(1); ptr_ += 1; return 0;
            case U16: require(2); ptr_ += 2; return 0;
            case I16: require(2); ptr_ += 2; return 0;
            case U32: {
                uint32_t v = read_u32();
                return v;
            }
            case I32: require(4); ptr_ += 4; return 0;
            case F32: require(4); ptr_ += 4; return 0;
            case BOOL: require(1); ptr_ += 1; return 0;
            case STRING: {
                (void)read_string();
                return 0;
            }
            case ARRAY: {
                uint32_t elem = read_u32();
                uint64_t n = read_u64();
                for (uint64_t i = 0; i < n; ++i) (void)read_metadata_value(elem);
                return 0;
            }
            case U64: return read_u64();
            case I64: require(8); ptr_ += 8; return 0;
            case F64: require(8); ptr_ += 8; return 0;
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
    std::unordered_map<std::string, uint64_t> metadata_u64_;
};
