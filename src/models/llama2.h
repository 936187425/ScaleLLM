#pragma once

#include <torch/torch.h>

#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/parallel_args.h"
#include "models/parameters.h"

// port LLAMA's model to C++ API:
// https://github.com/facebookresearch/llama/blob/main/llama/model.py
namespace llm::llama2 {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(uint32_t layer_id,
                const ModelArgs& args,
                const ParallelArgs& parallel_args,
                const torch::ScalarType& dtype,
                const torch::Device& device)
      : layer_id_(layer_id), parallel_args_(parallel_args) {
    const auto world_size = parallel_args.world_size();
    const int64_t dim = args.dim();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);

    n_local_heads_ = n_heads / world_size;
    n_local_kv_heads_ = n_kv_heads / world_size;
    head_dim_ = dim / n_heads;

    // register submodules
    // TODO: fuse wq, wk, wv into one linear layer
    wq_ = register_module("wq",
                          ColumnParallelLinear(dim,
                                               n_heads * head_dim_,
                                               /*gather_output=*/false,
                                               parallel_args,
                                               dtype,
                                               device));
    wk_ = register_module("wk",
                          ColumnParallelLinear(dim,
                                               n_kv_heads * head_dim_,
                                               /*gather_output=*/false,
                                               parallel_args,
                                               dtype,
                                               device));
    wv_ = register_module("wv",
                          ColumnParallelLinear(dim,
                                               n_kv_heads * head_dim_,
                                               /*gather_output=*/false,
                                               parallel_args,
                                               dtype,
                                               device));
    wo_ = register_module("wo",
                          RowParallelLinear(n_heads * head_dim_,
                                            dim,
                                            /*input_is_parallel=*/true,
                                            parallel_args,
                                            dtype,
                                            device));

    // initialize positional embedding
    // TODO: need to adjust the max_seq_len
    const auto rotary_dim = args.dim() / args.n_heads();
    pos_emb_ = register_module("pos_emb",
                               RotaryEmbedding(rotary_dim,
                                               args.max_seq_len(),
                                               /*scaling_factor=*/0.0f,
                                               /*interleaved=*/true,
                                               dtype,
                                               device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    const auto num_tokens = x.size(0);
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto query = wq_(x);
    auto key = wk_(x);
    auto value = wv_(x);

    // (num_tokens, n_local_heads, head_dim)
    query = query.view({num_tokens, n_local_heads_, head_dim_});
    key = key.view({num_tokens, n_local_kv_heads_, head_dim_});
    value = value.view({num_tokens, n_local_kv_heads_, head_dim_});

    // (num_tokens, n_local_heads, head_dim)
    // apply positional embedding
    std::tie(query, key) = pos_emb_(query, key, positions);

    // store k/v into cache based on slots
    kv_cache.set_kv_cache(input_params.slot_ids, key, value);

    auto output = torch::zeros_like(query);
    const auto num_prompt_tokens = input_params.num_prompt_tokens;
    if (num_prompt_tokens > 0) {
      // process sequences with prompt tokens (prefill)
      auto sliced_output =
          output.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      auto sliced_query =
          query.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      auto sliced_key =
          key.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      auto sliced_value =
          value.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      attention::varlen_masked_self_attention(sliced_query,
                                              sliced_key,
                                              sliced_value,
                                              input_params.cu_seq_lens,
                                              input_params.max_seq_len,
                                              sliced_output);
    }

    if (num_prompt_tokens < num_tokens) {
      // process sequences without prompt tokens (decode)
      auto sliced_output = output.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
      auto sliced_query = query.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
      attention::single_token_masked_self_attention(
          kv_cache,
          sliced_query,
          input_params.block_tables,
          input_params.context_lens,
          input_params.max_context_len,
          sliced_output);
    }
    output = output.contiguous().view({num_tokens, -1});
    return wo_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    wq_->load_state_dict(state_dict.select("wq."));
    wk_->load_state_dict(state_dict.select("wk."));
    wv_->load_state_dict(state_dict.select("wv."));
    wo_->load_state_dict(state_dict.select("wo."));
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear wq_{nullptr};

  ColumnParallelLinear wk_{nullptr};

  ColumnParallelLinear wv_{nullptr};

  RowParallelLinear wo_{nullptr};

  RotaryEmbedding pos_emb_{nullptr};

  uint32_t layer_id_;

  ParallelArgs parallel_args_;

  // configs used in forward
  int64_t n_local_heads_;
  int64_t n_local_kv_heads_;
  int64_t head_dim_;
};
TORCH_MODULE(Attention);

class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(const ModelArgs& args,
                  const ParallelArgs& parallel_args,
                  const torch::ScalarType& dtype,
                  const torch::Device& device) {
    const int64_t dim = args.dim();
    const int64_t multiple_of = args.multiple_of();
    const float ffn_dim_multiplier = args.ffn_dim_multiplier().value_or(1.0f);
    int64_t hidden_dim = 4 * dim;
    hidden_dim = 2 * hidden_dim / 3;
    // custom dim factor multiplier
    hidden_dim *= ffn_dim_multiplier;
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);

    // register the weight parameter
    // TODO: fuse w1 and w3 into one linear layer
    w1_ = register_module("w1",
                          ColumnParallelLinear(dim,
                                               hidden_dim,
                                               /*gather_output=*/false,
                                               parallel_args,
                                               dtype,
                                               device));
    w2_ = register_module("w2",
                          RowParallelLinear(hidden_dim,
                                            dim,
                                            /*input_is_parallel=*/true,
                                            parallel_args,
                                            dtype,
                                            device));
    w3_ = register_module("w3",
                          ColumnParallelLinear(dim,
                                               hidden_dim,
                                               /*gather_output=*/false,
                                               parallel_args,
                                               dtype,
                                               device));
  }

  torch::Tensor forward(torch::Tensor x) {
    namespace F = torch::nn::functional;
    return w2_(F::silu(w1_(x)) * w3_(x));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    w1_->load_state_dict(state_dict.select("w1."));
    w2_->load_state_dict(state_dict.select("w2."));
    w3_->load_state_dict(state_dict.select("w3."));
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear w1_{nullptr};
  RowParallelLinear w2_{nullptr};
  ColumnParallelLinear w3_{nullptr};
};
TORCH_MODULE(FeedForward);

class TransformerBlockImpl : public torch::nn::Module {
 public:
  TransformerBlockImpl(uint32_t layer_id,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       const torch::ScalarType& dtype,
                       const torch::Device& device)
      : layer_id_(layer_id), parallel_args_(parallel_args) {
    // register submodules
    attention_ = register_module(
        "attention", Attention(layer_id, args, parallel_args, dtype, device));
    feed_forward_ = register_module(
        "feed_forward", FeedForward(args, parallel_args, dtype, device));
    attention_norm_ = register_module(
        "attention_norm", RMSNorm(args.dim(), args.norm_eps(), dtype, device));
    ffn_norm_ = register_module(
        "ffn_norm", RMSNorm(args.dim(), args.norm_eps(), dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h =
        x + attention_(attention_norm_(x), positions, kv_cache, input_params);
    auto out = h + feed_forward_(ffn_norm_(h));
    return out;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attention_->load_state_dict(state_dict.select("attention."));
    feed_forward_->load_state_dict(state_dict.select("feed_forward."));
    attention_norm_->load_state_dict(state_dict.select("attention_norm."));
    ffn_norm_->load_state_dict(state_dict.select("ffn_norm."));
  }

 private:
  // parameter members, must be registered
  Attention attention_{nullptr};

  FeedForward feed_forward_{nullptr};

  RMSNorm attention_norm_{nullptr};

  RMSNorm ffn_norm_{nullptr};

  uint32_t layer_id_;

  ParallelArgs parallel_args_;
};
TORCH_MODULE(TransformerBlock);

class ModelImpl : public torch::nn::Module {
 public:
  ModelImpl(const ModelArgs& args,
            const ParallelArgs& parallel_args,
            const torch::ScalarType& dtype,
            const torch::Device& device)
      : parallel_args_(parallel_args) {
    // register submodules
    tok_embeddings_ = register_module(
        "tok_embeddings",
        ParallelEmbedding(
            args.vocab_size(), args.dim(), parallel_args, dtype, device));
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = TransformerBlock(i, args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_ = register_module(
        "norm", RMSNorm(args.dim(), args.norm_eps(), dtype, device));
    output_ = register_module("output",
                              ColumnParallelLinear(args.dim(),
                                                   args.vocab_size(),
                                                   /*gather_output=*/true,
                                                   parallel_args,
                                                   dtype,
                                                   device));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = tok_embeddings_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    h = norm_(h);
    // select last token for each sequence
    h = h.index_select(/*dim=*/0, input_params.last_token_indicies);
    auto logits = output_(h);
    return logits;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    tok_embeddings_->load_state_dict(state_dict.select("tok_embeddings."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("norm."));
    output_->load_state_dict(state_dict.select("output."));
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding tok_embeddings_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<TransformerBlock> layers_;

  RMSNorm norm_{nullptr};

  ColumnParallelLinear output_{nullptr};

  ParallelArgs parallel_args_;
};
TORCH_MODULE(Model);

}  // namespace llm::llama2