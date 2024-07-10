#pragma once
# include<glog/logging.h>
# include<torch/torch.h>
#include "layers/embedding.h"
# include "model_loader/state_dict.h"
# include "string.h"
namespace llm{


class ParallelLoRAImpl : public torch::nn::Module{
public:
    ParallelLoRAImpl()=default;
    ~ParallelLoRAImpl()=default;
    virtual torch::Tensor forward(torch::Tensor input) const=0;
    virtual void load_state_dict(const StateDict& state_dict)=0;
    virtual void verify_loaded_weights(const std::string& prefix="") const=0;

};


class ColumnParallelLinearLoRAImpl: public ParallelLoRAImpl{
public:
    ColumnParallelLinearLoRAImpl();
    torch::Tensor forward(torch::Tensor input) const override;
    void load_state_dict(const StateDict& StateDict) override;
    void verify_loaded_weights(const std::string& prefix="") const override;

private:

};
TORCH_MODULE(ColumnParallelLinearLoRA);


class RowParallelLinearLoRAImpl: public ParallelLoRAImpl{
public:
    RowParallelLinearLoRAImpl();
    torch::Tensor forward(torch::Tensor input) const override;
    void load_state_dict(const StateDict& StateDict) override;
    void verify_loaded_weights(const std::string& prefix="") const override;

private:
};
TORCH_MODULE(RowParallelLinearLoRA);




class ParallelEmbeddingLoRAImpl: public ParallelLoRAImpl{
public:
    ParallelEmbeddingLoRAImpl();
    torch::Tensor forward(torch::Tensor input) const override;
    void load_state_dict(const StateDict& StateDict) override;
    void verify_loaded_weights(const std::string& prefix="") const override;    
private:
    //  Parallel embedding
    ParallelEmbedding embedding_ = {nullptr};
    //  Parallel LoRA

    bool merge_weights_=true;
    int lora_alpha_ =1;
    int r_=0;

};
TORCH_MODULE(ParallelEmbeddingLoRA); 

}//namespace llm
