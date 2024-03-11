#include "handler.h"

#include <gflags/gflags.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <memory>

#include "flash_attn_handler.h"
#include "flash_infer_handler.h"
#include "ref_handler.h"

// decide which attention implementation to use
DEFINE_string(attention_handler,
              "auto",
              "attention handler, e.g. auto, pytorch, flash_attn, flash_infer");

namespace llm {

// create an attention handler with alibi slopes
std::unique_ptr<AttentionHandler> AttentionHandler::create_handler_with_alibi(
    const ModelArgs& args,
    const torch::Device& device,
    torch::optional<torch::Tensor> alibi_slopes) {
  const int64_t head_dim = args.hidden_size() / args.n_heads();
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // check if the user specified the attention handler
  if (boost::iequals(FLAGS_attention_handler, "pytorch")) {
    return std::make_unique<RefHandler>(scale, alibi_slopes);
  }
  if (boost::iequals(FLAGS_attention_handler, "flash_attn")) {
    CHECK(device.is_cuda()) << "flash_attn only supports cuda device";
    return std::make_unique<FlashAttnHandler>(scale, alibi_slopes);
  }
  if (boost::iequals(FLAGS_attention_handler, "flash_infer")) {
    CHECK(device.is_cuda()) << "flash_infer only supports cuda device";
    return std::make_unique<FlashInferHandler>(scale, alibi_slopes);
  }

  // choose the best handler based on device type
  if (device.is_cuda()) {
    // use flash_attn for cuda device
    return std::make_unique<FlashAttnHandler>(scale, alibi_slopes);
  }

  // use slower ref handler for other devices for now.
  return std::make_unique<RefHandler>(scale, alibi_slopes);
}

// create an attention handler with ROPE
std::unique_ptr<AttentionHandler> AttentionHandler::create_handler_with_rope(
    const ModelArgs& args,
    bool interleaved,
    torch::ScalarType dtype,
    const torch::Device& device) {
  const int64_t head_dim = args.hidden_size() / args.n_heads();
  // default to use head_dim if rotary_dim is not specified
  int64_t rotary_dim = args.rotary_dim() > 0 ? args.rotary_dim() : head_dim;
  // apply rotary_dim percentage
  rotary_dim = static_cast<int64_t>(rotary_dim * args.rotary_pct());

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // check if the user specified the attention handler
  if (boost::iequals(FLAGS_attention_handler, "pytorch")) {
    return std::make_unique<RefHandler>(scale,
                                        rotary_dim,
                                        args.max_position_embeddings(),
                                        args.rope_scaling(),
                                        args.rope_theta(),
                                        interleaved,
                                        dtype,
                                        device);
  }
  if (boost::iequals(FLAGS_attention_handler, "flash_attn")) {
    CHECK(device.is_cuda()) << "flash_attn only supports cuda device";
    return std::make_unique<FlashAttnHandler>(scale,
                                              rotary_dim,
                                              args.max_position_embeddings(),
                                              args.rope_scaling(),
                                              args.rope_theta(),
                                              interleaved,
                                              dtype,
                                              device);
  }
  if (boost::iequals(FLAGS_attention_handler, "flash_infer")) {
    CHECK(device.is_cuda()) << "flash_infer only supports cuda device";
    return std::make_unique<FlashInferHandler>(scale,
                                               rotary_dim,
                                               args.max_position_embeddings(),
                                               args.rope_scaling(),
                                               args.rope_theta(),
                                               interleaved,
                                               dtype,
                                               device);
  }

  // choose the best handler based on device type
  if (device.is_cuda()) {
    // use flash_attn for cuda device
    return std::make_unique<FlashAttnHandler>(scale,
                                              rotary_dim,
                                              args.max_position_embeddings(),
                                              args.rope_scaling(),
                                              args.rope_theta(),
                                              interleaved,
                                              dtype,
                                              device);
  }

  // use slower ref handler for other devices for now.
  return std::make_unique<RefHandler>(scale,
                                      rotary_dim,
                                      args.max_position_embeddings(),
                                      args.rope_scaling(),
                                      args.rope_theta(),
                                      interleaved,
                                      dtype,
                                      device);
}

}  // namespace llm
