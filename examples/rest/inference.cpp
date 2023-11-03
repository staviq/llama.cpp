#include "inference.hpp"

#include <sstream>

namespace LlamaREST
{

Model::Model()
{
    static float tensor_split[LLAMA_MAX_DEVICES] = {0};
    _model_params = {
        /*.n_gpu_layers                =*/ 99,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ tensor_split,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
    };
}

Model::Model(const std::string &filename, uint16_t cfg_slot)
:
    Model()
{
    std::string cfg_filename = filename + ".json";
    auto cfg = JsonFile(cfg_filename);
    if (cfg.content().empty())
    {
        auto & cfg_json = cfg.content();
        cfg_json["0"]["n_gpu_layers"] = _model_params.n_gpu_layers;
        cfg_json["0"]["main_gpu"]     = _model_params.main_gpu;
        cfg_json["0"]["tensor_split"] = json::array();
        cfg_json["0"]["use_mmap"]     = _model_params.use_mmap;
        cfg_json["0"]["use_mlock"]    = _model_params.use_mlock;
        cfg.save();
        load(filename, cfg.content()["0"]);
    }
    else
    {
        if (cfg.content().contains(std::to_string(cfg_slot)))
        {
            load(filename, cfg.content()[std::to_string(cfg_slot)]);
        }
        else
        {
            LOG_TEE("No such configuration slot in the model config file: '%d'\n", cfg_slot);
            load(filename, cfg.content()["0"]);
        }
    }

}

Model::~Model()
{
    const std::lock_guard<std::mutex> lock(_busy);

    if (_model)
    {
        llama_free_model(_model);
    }
}

bool Model::load(const std::string &filename, json params)
{
    const std::lock_guard<std::mutex> lock(_busy);

    if (_loaded)
    {
        LOG_TEE("This model is already loaded.\n");
        return false;
    }

    if (params.contains("n_gpu_layers")) {_model_params.n_gpu_layers = params["n_gpu_layers"].get<int32_t>();}
    if (params.contains("main_gpu"))     {_model_params.main_gpu     = params["main_gpu"].get<int32_t>();}
    if (params.contains("tensor_split")) {
        auto ts = params["tensor_split"].get<std::vector<float>>();
        memcpy(((float*)_model_params.tensor_split), ts.data(), ts.size() * sizeof(float));
    }
    if (params.contains("use_mmap"))     {_model_params.use_mmap     = params["use_mmap"].get<bool>();}
    if (params.contains("use_mlock"))    {_model_params.use_mlock    = params["use_mlock"].get<bool>();}

    _model = llama_load_model_from_file(filename.c_str(), _model_params);

    if (_model == NULL) {
        LOG_TEE("%s: error: unable to load model\n" , __func__);
        return _loaded = false;
    }

    _model_name = filename;
    return _loaded = true;
}

bool Model::loaded()
{
    return _loaded;
}

llama_model *Model::model()
{
    return _loaded ? _model : nullptr;
}

const std::string &Model::name()
{
    return _model_name;
}

Context::Context()
{
    const std::lock_guard<std::mutex> lock(_busy);
    _ctx_params = llama_context_default_params();
    //_ctx_params.seed = 1;
}

Context::Context( std::shared_ptr<Model> model, json params )
:
    Context()
{
    create(model,params);
}

Context::~Context()
{
    llama_free(_ctx);
}

bool Context::create_unsafe(std::shared_ptr<Model> model, json params, bool save)
{
    _jparams = params;
    _model = model;

    if (params.contains("n_ctx")) {_ctx_params.n_ctx = params["n_ctx"].get<uint32_t>();}

    _ctx_params.seed = time(NULL);

    if (!model->model())
    {
        LOG_TEE("Invalid model when creating context.\n");
        return _created = false;
    }

    _ctx = llama_new_context_with_model(model->model(), _ctx_params);

    if (_ctx == NULL) {
        LOG_TEE("%s: error: failed to create the llama_context\n" , __func__);
        return _created = false;
    }

    {
        LOG("Warming up the model with an empty run\n");

        std::vector<llama_token> tmp = { llama_token_bos(model->model()), llama_token_eos(model->model()), };
        llama_decode(_ctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) _ctx_params.n_batch), 0, 0));
        llama_kv_cache_tokens_rm(_ctx, -1, -1);
        llama_reset_timings(_ctx);
    }

    if (save) save_unsafe();
    //restore_unsafe();

    return _created = true;
}

bool Context::created()
{
    return _created;
}

llama_context *Context::context()
{
    return _created ? _ctx : nullptr;
}

void Context::save_unsafe()
{
    std::vector<uint8_t> state_mem(llama_get_state_size(_ctx));
    llama_copy_state_data(_ctx, state_mem.data());

    const char* const src = (const char*) state_mem.data();
    const int src_size = (int)(state_mem.size());
    const int max_dst_size = LZ4_compressBound(src_size);

    _state = (char*)malloc((size_t)max_dst_size);
    if (_state == NULL) {
        LOG_TEE("Failed to allocate memory for *compressed_data.\n");
        return;
    }

    _state_original_size = src_size;
    _state_compressed_size = LZ4_compress_fast(src, _state, src_size, max_dst_size, 1);

    if (_state_compressed_size <= 0) {
        LOG_TEE("A 0 or negative result from LZ4_compress_default() indicates a failure trying to compress the data.\n");
        return;
    }
    if (_state_compressed_size > 0) {
        LOG_TEE("State saved, ratio: %.2f\n", (float) _state_compressed_size/src_size);
    }
    
    _state = (char *)realloc(_state, (size_t)_state_compressed_size);
    if (_state == NULL) {
        LOG_TEE("Failed to re-alloc memory for compressed_data.\n");
        return;
    }
}

void Context::restore_unsafe()
{
    auto start = std::chrono::steady_clock::now();

    llama_free(_ctx);
    create_unsafe(_model,_jparams,false);

    std::vector<uint8_t> state_mem(_state_original_size);
    //FILE * fp_read = fopen("dump_state.bin.lz4", "rb");
    //const size_t ret = fread(state_mem.data(), 1, state_mem.size(), fp_read);

    const int src_size = (int)(state_mem.size());

    //const char* const compressed_data = (const char*) state_mem.data();
    //const int compressed_data_size = ret;

    char* const regen_buffer = (char*)malloc(src_size);
    if (regen_buffer == NULL) {
        LOG_TEE("Failed to allocate memory for *regen_buffer.\n");
        return;
    }
    const int decompressed_size = LZ4_decompress_safe(_state, regen_buffer, _state_compressed_size, src_size);
    if (decompressed_size < 0) {
        LOG_TEE("A negative result from LZ4_decompress_safe (%d) indicates a failure trying to decompress the data.  See exit code (echo $?) for value returned.\n", decompressed_size);
        return;
    }
    if (decompressed_size >= 0) {
        LOG_TEE("State restored\n");
    }
    if (decompressed_size != src_size) {
        LOG_TEE("Decompressed data is different from original!\n");
        return;
    }

    memcpy(state_mem.data(), regen_buffer, decompressed_size);
    free(regen_buffer);
    
    if ((size_t)decompressed_size != state_mem.size()) {
        LOG_TEE("Failed to read state\n");
        //llama_free(_ctx);
        //llama_free_model(model);
        //return 1;
        return;
    }

    llama_set_state_data(_ctx, state_mem.data());

    auto end = std::chrono::steady_clock::now();
    float timer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    LOG_TEE("State reset in %0.2fs\n", timer/1000.0);
}

Inference::Inference(CmdlArgs & cmdlargs, httplib::Server & srv, LLRestUuid & uuid_generator)
:
    _args(cmdlargs),
    _srv(srv),
    _uuidg(uuid_generator)
{
    llama_backend_init(false);
    const auto & params = _args.params();
    if (params.contains("model"))
    {
        std::string mdir = params["mdir"];
        std::string model = params["model"];
        if ( model.substr(0, mdir.length()) == mdir )
        {
            model_load( model, json( {{"cfgslot", 0}} ) );
        }
        else
        {
            model_load( mdir + "/" + model, json( {{"cfgslot", 0}} ) );
        }

        context_create(json( {{"n_ctx", 4096}} ));
    }

    _handlers.post["/tokenize/"] =
    _handlers.post["/tokenize"] = [&](const Request &req, Response &res) {
        const std::lock_guard<std::mutex> lock(_busy);
        json response;
        json content;

        bool fmt = req.has_param("fmt") ? ( req.get_param_value("fmt") == "true" || req.get_param_value("fmt") == "1" ? true : false ) : false;
        bool raw = req.has_param("raw") ? ( req.get_param_value("raw") == "true" || req.get_param_value("raw") == "1" ? true : false ) : false;

        if (req.get_header_value("Content-Type") == "application/json")
        {
            content = req.body.empty() ? json() : json::parse(req.body);
        }
        else
        {
            content["text"] = req.body.empty() ? "" : req.body;
        }

        auto result = raw ? tokenize_raw_unsafe(content) : tokenize_unsafe(content);
        if (result.first) {
            response["result"] = result.second;
            response["length"] = result.second.size();
        } else {
            response["error"] = "Failed to tokenize.";
            response["length"] = 0;
        }

        res.set_content( fmt ? response.dump(4, ' ') : response.dump(), "application/json");
    };

    _handlers.post["/detokenize/"] =
    _handlers.post["/detokenize"] = [&](const Request &req, Response &res) {
        const std::lock_guard<std::mutex> lock(_busy);
        json response;
        json content;

        bool fmt = req.has_param("fmt") ? ( req.get_param_value("fmt") == "true" || req.get_param_value("fmt") == "1" ? true : false ) : false;

        if (req.get_header_value("Content-Type") == "application/json")
        {
            try {
                content = req.body.empty() ? json() : json::parse(req.body);
            } catch (std::exception &e) {
                LOG_TEE("Exception: %s\n", e.what());
                content = json();
            }
        }
        else
        {
            response["error"] = "Invalid Content-Type, expected 'application/json' received '" + req.get_header_value("Content-Type") + "'";
        }

        auto result = detokenize_unsafe(content);
        if (result.first) {
            response["result"] = result.second;
            response["length"] = result.second.size();
        } else {
            response["error"] += "Failed to tokenize.";
            response["length"] = 0;
        }

        res.set_content( fmt ? response.dump(4, ' ') : response.dump(), "application/json");
    };

    _handlers.post["/completion/"] =
    _handlers.post["/completion"] = [&](const Request &req, Response &res) {
        const std::lock_guard<std::mutex> lock(_busy);
        json response;
        json content;

        bool fmt = req.has_param("fmt") ? ( req.get_param_value("fmt") == "true" || req.get_param_value("fmt") == "1" ? true : false ) : false;
        bool raw = req.has_param("raw") ? ( req.get_param_value("raw") == "true" || req.get_param_value("raw") == "1" ? true : false ) : false;

        const auto & ctheader = req.has_header("Content-Type") ? req.get_header_value("Content-Type") : "";
        const auto & ctype = ctheader.substr(0, ctheader.find(";"));
        const auto & cencoding = ctheader.substr(ctype.length());

        if (ctype == "application/json") {
            auto result = completion_unsafe( req.body.empty() ? json() : json::parse(req.body) );

            if (result.first) {
                response["result"] = result.second;
                response["length"] = result.second.size();
            } else {
                response["error"] += "Failed to tokenize.";
                response["length"] = 0;
            }
        } else if (ctype == "text/plain") {
            auto result = completion_unsafe( json( {{"text", req.body.empty() ? "" : req.body},{"n", 16}} ) );

            if (result.first) {
                response["result"] = result.second;
                response["length"] = result.second.size();
            } else {
                response["error"] += "Failed to tokenize.";
                response["length"] = 0;
            }
        } else {
            response["error"] = "Invalid Content-Type, expected 'application/json' or 'text/plain' received '" + req.get_header_value("Content-Type") + "'";
        }

        res.set_content( fmt ? response.dump(4, ' ') : response.dump(), "application/json");
    };

    srv_register_handlers(srv, _handlers);
}

bool Inference::model_load_unsafe(std::string model, json params)
{
    auto start = std::chrono::steady_clock::now();

    model_unload_unsafe();

    if (params.empty())
    {
        _model.reset( new Model(model) );
    }
    else if (params.contains("cfgslot"))
    {
        _model.reset( new Model(model, params["cfgslot"].get<uint32_t>()) );
    }
    else
    {
        _model.reset( new Model() );
        _model->load( model, params );
    }

    auto end = std::chrono::steady_clock::now();
    double timer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (! _model->loaded())
    {
        _model.reset();
        return false;
    }

    LOG_TEE("Model loaded in %0.3fs\n", timer/1000.0);

    return true;

    // _model_params.model = model;

    // std::tie(_model, _ctx) = llama_init_from_gpt_params(_model_params);
    // if (_model == nullptr || _ctx == nullptr) {
    //     fprintf(stderr, "%s : failed to init\n", __func__);
    //     return false;
    // }

    // return true;
}

void Inference::model_unload_unsafe()
{
    if (_context) {_context.reset();}
    if (_model)   {_model.reset();}
}

bool Inference::context_create_unsafe(json params)
{
    context_destroy_unsafe();

    if (params.empty())
    {
        _context.reset( new Context(_model) );
        return _context && _context->created();
    }
    else
    {
        _context.reset( new Context(_model, params) );
        return _context && _context->created();
    }

    return false;
}

void Inference::context_destroy_unsafe()
{
    if (_context) {_context.reset();}
}

std::pair<bool,json> Inference::tokenize_unsafe(json input)
{
    if(!_model || !_model->loaded()) {
        LOG_TEE("No model loaded.\n");
        return {false,json()};
    }
    if(!_context || !_context->created()) {
        LOG_TEE("No context created.\n");
        return {false,json()};
    }

    auto tokens_list = llama_tokenize(_context->context(), input["text"], true);

    return {true,tokens_list};
}

std::pair<bool,json> Inference::tokenize_raw_unsafe(json input)
{
    if(!_model || !_model->loaded()) {
        LOG_TEE("No model loaded.\n");
        return {false,json()};
    }
    if(!_context || !_context->created()) {
        LOG_TEE("No context created.\n");
        return {false,json()};
    }

    auto tokens_list = llama_tokenize(_context->context(), input["text"], false, true);

    return {true,tokens_list};
}

std::pair<bool,std::string> Inference::detokenize_unsafe(json input)
{
    if(!_model || !_model->loaded()) {
        LOG_TEE("No model loaded.\n");
        return {false,std::string()};
    }
    if(!_context || !_context->created()) {
        LOG_TEE("No context created.\n");
        return {false,std::string()};
    }

    std::vector<char> cache(8, 0);
    std::string result;
    uint8_t first = 0;
    //LOG_TEE("%s\n",input.dump(1,'\t').c_str());
    for( const auto & piece: input)
    {
        cache.clear();
        const llama_token & token = piece;
        int n_tokens = llama_token_to_piece(_model->model(), token, cache.data(), cache.size());
        if (n_tokens < 0) {
            //cache.resize(-n_tokens);
            cache.resize( (-n_tokens) * 2 );
            int check = llama_token_to_piece(_model->model(), token, cache.data(), cache.size());
            if (check != -n_tokens) {
                LOG_TEE("Detokenize failed.\n");
                return {false,""};
            }
            n_tokens = check;
        } else {
            //cache.resize(cache.capacity() > 0 ? cache.capacity() * 2 : 16);
        }

        if (token == llama_token_bos(_model->model())) {
            first++;
        }

        if( first == 1 && token != llama_token_bos(_model->model()) && n_tokens > 0 && cache.at(0) == ' ' ) {
            result += std::string(cache.data()+1, n_tokens-1);
            first++;
        } else {
            result += std::string(cache.data(), n_tokens);
        }
    }

    return {true,result};
}

std::pair<bool, std::string> Inference::completion_unsafe(json input)
{
    if(!_model || !_model->loaded()) {
        LOG_TEE("No model loaded.\n");
        return {true,json( {{"error", "No model loaded."}} )};
    }
    if(!_context || !_context->created()) {
        LOG_TEE("No context created.\n");
        return {true,json( {{"error", "No context created."}} )};
    }

    //_context->save();
    _context->restore();

    //context_create_unsafe();

    auto ctx = _context->context();
    auto model = _model->model();

    std::string prompt = input["text"];
    size_t n_predict = input["n"];

    // number of parallel batches
    int n_parallel = 1;
    // total length of the sequences including the prompt
    int n_len = 32;

    std::string result0;

    // tokenize prompt
    auto tokens = llama_tokenize(ctx, prompt, true);

    n_len = tokens.size() + n_predict;

    auto start = std::chrono::steady_clock::now();
    int tpscnt = 0;

    // create a llama_batch
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(std::max(tokens.size(), (size_t)n_parallel), 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_batch_add(batch, tokens[i], i, { 0 }, false);
    }
    //GGML_ASSERT(batch.n_tokens == (int) tokens_list.size());

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        //return 1;
    }

    // assign the system KV cache to all parallel sequences
    // this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    for (int32_t i = 1; i < n_parallel; ++i) {
        llama_kv_cache_seq_cp(ctx, 0, i, 0, batch.n_tokens);
    }

    if (n_parallel > 1) {
        LOG_TEE("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
    }

    // main loop

    // we will store the parallel decoded sequences in this vector
    std::vector<std::string> streams(n_parallel);

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_len) {
        // prepare the next batch
        llama_batch_clear(batch);

        // sample the next token for each parallel sequence / stream
        for (int32_t i = 0; i < n_parallel; ++i) {
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, i_batch[i]);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            const int   top_k = 40;
            const float top_p = 0.9f;
            const float temp  = 1.1f;

            llama_sample_top_k(ctx, &candidates_p, top_k, 1);
            llama_sample_top_p(ctx, &candidates_p, top_p, 1);
            llama_sample_temp (ctx, &candidates_p, temp);

            const llama_token new_token_id = llama_sample_token(ctx, &candidates_p);

            //const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of stream? -> mark the stream as finished
            if (new_token_id == llama_token_eos(model) || n_cur == n_len) {
                i_batch[i] = -1;
                LOG_TEE("\n");
                if (n_parallel > 1) {
                    LOG_TEE("%s: stream %d finished at n_cur = %d", __func__, i, n_cur);
                }

                continue;
            }

            // if there is only one stream, we print immediately to stdout
            if (n_parallel == 1) {
                LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str());
                fflush(stdout);
            }

            streams[i] += llama_token_to_piece(ctx, new_token_id);

            i_batch[i] = batch.n_tokens;

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { i }, true);

            n_decode += 1;
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            //return 1;
        }
    }

    LOG_TEE("\n");

    if (n_parallel > 1) {
        LOG_TEE("\n");

        for (int32_t i = 0; i < n_parallel; ++i) {
            LOG_TEE("sequence %d:\n\n%s%s\n\n", i, prompt.c_str(), streams[i].c_str());
        }
    }

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    //_context->restore();

    return {true,json( streams[0] )};
}

} // namespace LlamaREST
