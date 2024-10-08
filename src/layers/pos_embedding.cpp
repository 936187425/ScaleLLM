#include "pos_embedding.h"

#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>

#include "common/slice.h"
#include "kernels/pos_embedding_kernels.h"

DEFINE_bool(disable_custom_kernels, false, "disable all custom kernels");

namespace llm {

namespace {
using torch::indexing::None;
using ISlice = torch::indexing::Slice;

// [1, 2, 3, 4] => [-2, 1, -4, 3]
inline torch::Tensor rotate_every_two(const torch::Tensor& x) {
  auto x1 = x.index({ISlice(), ISlice(), ISlice(0, None, 2)});
  auto x2 = x.index({ISlice(), ISlice(), ISlice(1, None, 2)});
  return torch::stack({-x2, x1}, /*dim=*/-1).flatten(/*start_dim=*/-2);
}

// apply interleaved rotary positional embedding
inline std::tuple<torch::Tensor, torch::Tensor>
apply_interleaved_rotary_pos_emb(const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& cos,
                                 const torch::Tensor& sin) {
  auto q_embed = (q * cos) + (rotate_every_two(q) * sin);
  auto k_embed = (k * cos) + (rotate_every_two(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

// [1, 2, 3, 4] => [-3, -4, 1, 2]
inline torch::Tensor rotate_half(const torch::Tensor& x) {
  auto chunks = x.chunk(2, /*dim=*/-1);
  return torch::cat({-chunks[1], chunks[0]}, /*dim=*/-1);
}

// apply rotary positional embedding
inline std::tuple<torch::Tensor, torch::Tensor> apply_rotated_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  auto q_embed = (q * cos) + (rotate_half(q) * sin);
  auto k_embed = (k * cos) + (rotate_half(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

// create right instance based on params
std::shared_ptr<RotaryEmbeddingImpl> create(
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    torch::Tensor inv_freq,
    bool interleaved,
    const torch::TensorOptions& options) {
  if (options.device().is_cuda() && !FLAGS_disable_custom_kernels) {
    // use custom kernels
    return std::make_shared<RotaryEmbeddingKernel>(
        rotary_dim, max_position_embeddings, inv_freq, interleaved, options);
  }
  return std::make_shared<RotaryEmbeddingGeneric>(
      rotary_dim, max_position_embeddings, inv_freq, interleaved, options);
}
}  // namespace

namespace detail {
// compute the inverse frequencies
// returns float32 tensor with shape [rotary_dim / 2]
torch::Tensor compute_default_inv_freq(int64_t rotary_dim, float theta) {
  CHECK(rotary_dim % 2 == 0) << "rotary_dim must be even";
  const auto slice = torch::arange(0, rotary_dim, 2, torch::kFloat32);
  return 1.0 / torch::pow(theta, slice / rotary_dim);
}

torch::Tensor apply_llama3_rope_scaling(torch::Tensor inv_freq,
                                        float factor,
                                        float low_freq_factor,
                                        float high_freq_factor,
                                        int64_t old_context_len) {
  Slice<float> inv_freq_slice(inv_freq.mutable_data_ptr<float>(),
                              inv_freq.numel());
  std::vector<float> new_inv_freq;
  new_inv_freq.reserve(inv_freq_slice.size());

  const float low_freq_wavelen = old_context_len / low_freq_factor;
  const float high_freq_wavelen = old_context_len / high_freq_factor;
  for (auto freq : inv_freq_slice) {
    float new_freq = freq;
    const float wavelen = 2 * M_PI / freq;
    if (wavelen < high_freq_wavelen) {
      // do nothing
    } else if (wavelen > low_freq_wavelen) {
      new_freq = freq / factor;
    } else {
      CHECK_NE(low_freq_wavelen, high_freq_wavelen);
      const float smooth = (old_context_len / wavelen - low_freq_factor) /
                           (high_freq_factor - low_freq_factor);
      new_freq = (1 - smooth) * freq / factor + smooth * freq;
    }
    new_inv_freq.push_back(new_freq);
  }
  return torch::tensor(new_inv_freq, inv_freq.options());
}

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos_sin,
    bool interleaved) {
  const auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  if (interleaved) {
    return apply_interleaved_rotary_pos_emb(q, k, chunks[0], chunks[1]);
  }
  return apply_rotated_rotary_pos_emb(q, k, chunks[0], chunks[1]);
}

}  // namespace detail

RotaryEmbedding::RotaryEmbedding(int64_t rotary_dim,
                                 int64_t max_position_embeddings,
                                 torch::Tensor inv_freq,
                                 bool interleaved,
                                 const torch::TensorOptions& options)
    : ModuleHolder(create(rotary_dim,
                          max_position_embeddings,
                          inv_freq,
                          interleaved,
                          options)) {}

RotaryEmbeddingGeneric::RotaryEmbeddingGeneric(
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    torch::Tensor inv_freq,
    bool interleaved,
    const torch::TensorOptions& options)
    : rotary_dim_(rotary_dim), interleaved_(interleaved) {
  // [max_position_embeddings]
  auto t = torch::arange(0, max_position_embeddings, 1, torch::kFloat32);
  // [max_position_embeddings, rotary_dim/2]
  const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
  // Create cos and sin embeddings.
  torch::Tensor emd;
  if (interleaved) {
    // [a, b, c, d] => [a, a, b, b, c, c, d, d]
    emd = freqs.repeat_interleave(/*repeats=*/2, /*dim=*/-1);
  } else {
    // [a, b, c, d] => [a, b, c, d, a, b, c, d]
    emd = torch::cat({freqs, freqs}, /*dim=*/-1);
  }

  const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin.to(options));
}

// inplace rotary positional embedding
std::tuple<torch::Tensor, torch::Tensor> RotaryEmbeddingGeneric::forward(
    const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions  // [num_tokens]
) const {
  DCHECK_GE(query.size(-1), rotary_dim_);
  auto query_rotary = query.index({"...", ISlice(0, rotary_dim_)});
  auto query_pass = query.index({"...", ISlice(rotary_dim_, None)});
  auto key_rotary = key.index({"...", ISlice(0, rotary_dim_)});
  auto key_pass = key.index({"...", ISlice(rotary_dim_, None)});

  namespace F = torch::nn::functional;
  auto cos_sin = F::embedding(positions, cos_sin_cache_);
  // add a new dimension for n_heads
  cos_sin = cos_sin.unsqueeze(1);
  std::tie(query_rotary, key_rotary) = detail::apply_rotary_pos_emb(
      query_rotary, key_rotary, cos_sin, interleaved_);
  return std::make_tuple(torch::cat({query_rotary, query_pass}, /*dim=*/-1),
                         torch::cat({key_rotary, key_pass}, /*dim=*/-1));
}

RotaryEmbeddingKernel::RotaryEmbeddingKernel(
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    torch::Tensor inv_freq,
    bool interleaved,
    const torch::TensorOptions& options)
    : rotary_dim_(rotary_dim), interleaved_(interleaved) {
  // [max_position_embeddings]
  auto t = torch::arange(0, max_position_embeddings, 1, torch::kFloat32);
  // [max_position_embeddings, rotary_dim/2]
  const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});

  const auto cos_sin = torch::cat({freqs.cos(), freqs.sin()}, /*dim=*/-1);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin.to(options));
}

// inplace rotary positional embedding
std::tuple<torch::Tensor, torch::Tensor> RotaryEmbeddingKernel::forward(
    const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions  // [num_tokens]
) const {
  DCHECK_GE(query.size(-1), rotary_dim_);
  torch::Tensor _query = query;
  torch::Tensor _key = key;
  kernel::apply_rotary_pos_emb(_query,
                               _key,
                               positions,
                               cos_sin_cache_,
                               static_cast<int>(rotary_dim_),
                               interleaved_);
  return std::make_tuple(query, key);
}

}  // namespace llm
