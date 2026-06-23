#pragma once
#include "tensor_view.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cstring>

#include <mio/mio.hpp>
#include <nlohmann/json.hpp>

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path) {
        std::error_code ec;
        mmap_.map(path, ec);
        if (ec) throw std::runtime_error("mmap failed on " + path + ": " + ec.message());

        const char* base = mmap_.data();
        size_t      file_size = mmap_.size();

        if (file_size < 8)
            throw std::runtime_error("safetensors file too small");

        uint64_t header_len = 0;
        std::memcpy(&header_len, base, 8);

        if (8 + header_len > file_size)
            throw std::runtime_error("safetensors header_len exceeds file size");

        auto header = nlohmann::json::parse(base + 8, base + 8 + header_len);

        const char* data_start = base + 8 + header_len;

        for (auto& [name, meta] : header.items()) {
            if (name == "__metadata__") continue;

            TensorView tv;
            tv.dtype = meta["dtype"].get<std::string>();
            tv.shape = meta["shape"].get<std::vector<int64_t>>();

            auto offsets = meta["data_offsets"].get<std::vector<uint64_t>>();
            uint64_t byte_start = offsets[0];
            uint64_t byte_end   = offsets[1];
            tv.nbytes = byte_end - byte_start;
            tv.data   = data_start + byte_start;

            tensors_[name] = std::move(tv);
        }
    }

    const TensorView& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("safetensors: tensor not found: " + name);
        return it->second;
    }

    bool has(const std::string& name) const {
        return tensors_.count(name) > 0;
    }

    size_t num_tensors() const { return tensors_.size(); }

    const std::unordered_map<std::string, TensorView>& all() const { return tensors_; }

private:
    mio::mmap_source mmap_;
    std::unordered_map<std::string, TensorView> tensors_;
};
