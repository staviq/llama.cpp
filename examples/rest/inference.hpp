#ifndef LLAMA_CPP_REST_INFERENCE_HPP
#define LLAMA_CPP_REST_INFERENCE_HPP

#include "libs.hpp"
#include "utils.hpp"
#include "cmdlargs.hpp"

#include "common.h"
#include "llama.h"

#include <mutex>
#include <vector>
#include <utility>

namespace LlamaREST
{

using json = nlohmann::json;

class Model
{
    public:
        Model();
        Model( const std::string & filename, uint16_t cfg_slot = 0 );
        ~Model();

    public:
        bool load(const std::string & filename, json params = json());
        bool loaded();

        const llama_model_params * const params() const {return &_model_params;}
        llama_model * model();

        const std::string & name();

    private:
        std::mutex _busy;
        std::atomic<bool> _loaded{false};

        std::string _model_name;

        llama_model_params _model_params;
        llama_model * _model;
};

class Context
{
    public:
        Context();
        Context( std::shared_ptr<Model> model, json params = json() );
        ~Context();

    public:
        bool create(std::shared_ptr<Model> model, json params = json()) {
            const std::lock_guard<std::mutex> lock(_busy);
            return create_unsafe(model,params);
        }
        bool created();

        const llama_context_params * const params() const {return &_ctx_params;}
        llama_context * context();
    public:
        void save() {
            const std::lock_guard<std::mutex> lock(_busy);
            save_unsafe();
        }
        void restore() {
            const std::lock_guard<std::mutex> lock(_busy);
            restore_unsafe();
        }
    private:
        bool create_unsafe(std::shared_ptr<Model> model, json params = json(), bool save = true);
        void save_unsafe();
        void restore_unsafe();

    private:
        std::mutex _busy;
        std::atomic<bool> _created{false};

        llama_context_params _ctx_params;
        llama_context * _ctx;

        json _jparams;
        std::shared_ptr<Model> _model = nullptr;

        char * _state = nullptr;
        size_t _state_original_size = 0;
        size_t _state_compressed_size = 0;
};

class Inference
{
    public:
        //Inference( json config ){}
        Inference(CmdlArgs & cmdlargs, httplib::Server & srv, LLRestUuid & uuid_generator);
        ~Inference(){}
    private:
        Inference(Inference & other) = delete;
        void operator=(const Inference &)  = delete;

    public:
        bool model_load(std::string model, json params = json()) {
            const std::lock_guard<std::mutex> lock(_busy);
            return model_load_unsafe(model, params);
        }

        void model_unload() {
            const std::lock_guard<std::mutex> lock(_busy);
            model_unload_unsafe();
        }

        bool context_create(json params = json()) {
            const std::lock_guard<std::mutex> lock(_busy);
            return context_create_unsafe(params);
        }

        void context_destroy() {
            const std::lock_guard<std::mutex> lock(_busy);
            context_destroy_unsafe();
        }

        std::pair<bool,json> tokenize(json input) {
            const std::lock_guard<std::mutex> lock(_busy);
            return tokenize_unsafe(input);
        }

        std::pair<bool,json> tokenize_raw(json input) {
            const std::lock_guard<std::mutex> lock(_busy);
            return tokenize_raw_unsafe(input);
        }

        std::pair<bool,std::string> detokenize(json input) {
            const std::lock_guard<std::mutex> lock(_busy);
            return detokenize_unsafe(input);
        }

        std::pair<bool,std::string> completion(json input) {
            const std::lock_guard<std::mutex> lock(_busy);
            return completion_unsafe(input);
        }

        std::shared_ptr<Model>   model()   {return _model;}
        std::shared_ptr<Context> context() {return _context;}

    private:
        bool model_load_unsafe(std::string model, json params = json());
        void model_unload_unsafe();
        bool context_create_unsafe(json params = json());
        void context_destroy_unsafe();
        std::pair<bool,json> tokenize_unsafe(json input);
        std::pair<bool,json> tokenize_raw_unsafe(json input);
        std::pair<bool,std::string> detokenize_unsafe(json input);
        std::pair<bool,std::string> completion_unsafe(json input);

    private:
        std::mutex _busy;

        CmdlArgs & _args;
        httplib::Server & _srv;
        LLRestUuid & _uuidg;

        std::shared_ptr<Model>   _model  {nullptr};
        std::shared_ptr<Context> _context{nullptr};

        SrvHandlers _handlers;
};

} // namespace LlamaREST

#endif