#pragma once

#include "utils.hpp"
#include "cmdlargs.hpp"
#include "libs.hpp"

namespace LlamaREST
{

class System
{
    public:
        System( CmdlArgs & cmdlargs, httplib::Server & srv, LLRestUuid & uuid_generator )
        :
            _cmdlargs(cmdlargs),
            _srv(srv),
            _uuid_generator(uuid_generator)
        {
            _srv.Get("/system/models", [&](const Request &req, Response &res) {
                json response;
                response["session"] = _uuid_generator.make();
                res.set_content( response.dump(), "application/json");
            });
        }

    private:
        std::vector<std::string> _ls( const std::string & path )
        {
            std::vector<std::string> results;
        }
    private:
        CmdlArgs & _cmdlargs;
        httplib::Server & _srv;
        LLRestUuid & _uuid_generator;
};

} // namespace LlamaREST
