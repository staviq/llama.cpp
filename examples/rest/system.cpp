#include "system.hpp"

namespace LlamaREST
{
System::System(CmdlArgs &cmdlargs, httplib::Server &srv, LLRestUuid &uuid_generator, Inference &infr)
:
    _args(cmdlargs),
    _srv(srv),
    _uuidg(uuid_generator),
    _infr(infr)
{
    _handlers.get["/system"] = [&](const Request &req, Response &res) {
        res.set_content("System", "text/plain");
    };

    _handlers.get["/system/models/:mdl"] = [&](const Request &req, Response &res) {
        auto mdl = req.path_params.at("mdl");
        res.set_content(mdl, "text/plain");
    };

    _handlers.post["/system/models/:mdl/load"] = [&](const Request &req, Response &res) {
        auto mdl = req.path_params.at("mdl");
        
        json response;
        response[mdl] = _infr.model_load( cmdlargs.params()["mdir"].get<std::string>().append("/").append(mdl), json() );

        res.set_content(response.dump(), "application/json");
    };

    _handlers.get["/system/models"] = [&](const Request &req, Response &res) {
        auto params = _args.params();
        json response;

        response["session"] = _uuidg.make();
        //response["models"] = json::array();

        const auto tmp_ls = _ls_m(params["mdir"]);
        for (const auto & m: tmp_ls)
        {
            const auto & cfg = JsonFile( std::string(m).append(".json") ).content();
            response["models"][m]["config"] = cfg;
        }
        
        res.set_content( response.dump(), "application/json");
    };

    srv_register_handlers(srv, handlers());
}

std::vector<std::string> System::_ls_m(const std::string &path)
{
    auto ls = _ls(path);
    std::vector<std::string> results;
    for( const auto & m: ls)
    {
        static const std::string ext = ".gguf";
        if ( m.substr( m.length() - ext.length() ) == ext )
        {
            results.push_back( m );
        }
    }

    return results;
}

std::vector<std::string> System::_ls(const std::string &path)
{
    std::vector<std::string> results;

#ifndef _WIN32
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (path.c_str())) != nullptr)
    {
        while ((ent = readdir (dir)) != nullptr)
        {
            const std::string name = ent->d_name;
            const auto dots = std::set<std::string>({".",".."});
            if ( dots.find( name ) == dots.end() )
            {
                results.push_back( ent->d_name );
            }
        }
        closedir (dir);
    }
#endif

    return results;
}

} // namespace LlamaREST
