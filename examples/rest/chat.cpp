#include "chat.hpp"

namespace LlamaREST
{
    
Chat::Chat(CmdlArgs &cmdlargs, httplib::Server &srv, LLRestUuid &uuid_generator, Inference &infr)
:
    _args(cmdlargs),
    _srv(srv),
    _uuidg(uuid_generator),
    _infr(infr)
{
        _handlers.get["/session/:sid"] = [&](const Request &req, Response &res) {
        const std::lock_guard<std::mutex> lock(_busy);
        json response;
        json content;

        auto _model   = _infr.model();
        auto _context = _infr.context();

        const auto & sid = ( req.path_params.find("sid") != req.path_params.end() ) ? req.path_params.at("sid") : "";
        bool fmt = req.has_param("fmt") ? ( req.get_param_value("fmt") == "true" || req.get_param_value("fmt") == "1" ? true : false ) : false;

        if(!_model || !_model->loaded()) {
            LOG_TEE("No model loaded.\n");
            response["error"] += "No model loaded.";
            response["length"] = 0;
            res.set_content( fmt ? response.dump(4, ' ') : response.dump(), "application/json");
            return;
        }
        if(!_context || !_context->created()) {
            LOG_TEE("No context created.\n");
            response["error"] += "No context created.";
            response["length"] = 0;
            res.set_content( fmt ? response.dump(4, ' ') : response.dump(), "application/json");
            return;
        }

        if (sid.empty())
        {
            response["sid"] = uuid_generator.make();
            response["session"] = {
                {"sid", response["sid"]},
                {"model", _model->name()},
                {"context", _context->params()->n_ctx},
                {"seed", _context->params()->seed}
            };
        }
        else
        {
            response["test"] = sid;
        }

        res.set_content( fmt ? response.dump(4, ' ') : response.dump(), "application/json");
    };
    
    _handlers.get["/session/"] = _handlers.get["/session/:sid"];
    _handlers.get["/session"] = _handlers.get["/session/:sid"];

    srv_register_handlers(srv, _handlers);
}

} // namespace LlamaREST
