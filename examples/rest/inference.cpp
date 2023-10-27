#include "inference.hpp"

namespace LlamaREST
{

bool Inference::model_load(std::string model, json params)
{
    const std::lock_guard<std::mutex> lock(_busy);
    
    _model_params.model = model;

    std::tie(_model, _ctx) = llama_init_from_gpt_params(_model_params);
    if (_model == nullptr || _ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return false;
    }

    return true;
}

} // namespace LlamaREST
