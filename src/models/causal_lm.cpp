#include "causal_lm.h"

#include <torch/torch.h>
#include <glog/logging.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <vector>

#include "args.h"
#include "huggingface/gpt2.h"
#include "huggingface/gpt_neox.h"
#include "huggingface/llama.h"
#include "huggingface/mistral.h"
#include "huggingface/aquila.h"
#include "input_parameters.h"
#include "llama.h"
#include "memory/kv_cache.h"
#include "model_loader/state_dict.h"

namespace llm {

std::unique_ptr<CausalLM> CausalLM::create(const ModelArgs& args,
                                           const QuantizationArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           torch::ScalarType dtype,
                                           const torch::Device& device) {
  // create models based on model type
  if (boost::iequals(args.model_type(), "llama2")) {
    LlamaModel llama2(args, quant_args, parallel_args, dtype, device);
    llama2->eval();
    return std::make_unique<llm::CausalLMImpl<LlamaModel>>(std::move(llama2));
  }
  // llama from hf models
  if (boost::iequals(args.model_type(), "llama")) {
    hf::LlamaModel llama2(args, quant_args, parallel_args, dtype, device);
    // set the module in evaluation/inference mode
    llama2->eval();
    return std::make_unique<llm::CausalLMImpl<hf::LlamaModel>>(
        std::move(llama2));
  }
  if (boost::iequals(args.model_type(), "gpt2")) {
    hf::GPT2Model gpt2(args, quant_args, parallel_args, dtype, device);
    // set the module in evaluation/inference mode
    gpt2->eval();
    return std::make_unique<llm::CausalLMImpl<hf::GPT2Model>>(std::move(gpt2));
  }
  if (boost::iequals(args.model_type(), "gpt_neox")) {
    hf::GPTNeoXModel gpt_neox(args, quant_args, parallel_args, dtype, device);
    // set the module in evaluation/inference mode
    gpt_neox->eval();
    return std::make_unique<llm::CausalLMImpl<hf::GPTNeoXModel>>(
        std::move(gpt_neox));
  }
  if (boost::iequals(args.model_type(), "mistral")) {
    hf::MistralModel mistral(args, quant_args, parallel_args, dtype, device);
    // set the module in evaluation/inference mode
    mistral->eval();
    return std::make_unique<llm::CausalLMImpl<hf::MistralModel>>(
        std::move(mistral));
  }
  if (boost::iequals(args.model_type(), "aquila")) {
    hf::AquilaModel aquila(args, quant_args, parallel_args, dtype, device);
    // set the module in evaluation/inference mode
    aquila->eval();
    return std::make_unique<llm::CausalLMImpl<hf::AquilaModel>>(
        std::move(aquila));
  }

  LOG(ERROR) << "Unsupported model type: " << args.model_type();
  return nullptr;
}

}  // namespace llm
