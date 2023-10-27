#ifndef LLAMA_CPP_REST_INFERENCE_HPP
#define LLAMA_CPP_REST_INFERENCE_HPP

#include "libs.hpp"

#include "common.h"
#include "llama.h"

#include <mutex>

namespace LlamaREST
{

using json = nlohmann::json;

class Inference
{
    public:
        //Inference( json config ){}
        Inference(){}
        ~Inference(){}
    private:
        Inference(Inference & other) = delete;
        void operator=(const Inference &)  = delete;

    public:
        bool model_load( std::string model, json params );

    private:
        std::mutex _busy;

        gpt_params _model_params;
        llama_model * _model = nullptr;
        llama_context * _ctx = nullptr;
};

} // namespace LlamaREST

#endif