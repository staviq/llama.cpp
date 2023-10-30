#ifndef LLAMA_CPP_REST_SYSTEM_HPP
#define LLAMA_CPP_REST_SYSTEM_HPP

#include "utils.hpp"
#include "cmdlargs.hpp"
#include "libs.hpp"
#include "inference.hpp"

#ifndef _WIN32
#include <dirent.h>
#endif

namespace LlamaREST
{

class System
{
    public:
        System( CmdlArgs & cmdlargs, httplib::Server & srv, LLRestUuid & uuid_generator, Inference & infr );

    private:
        std::vector<std::string> _ls_m( const std::string & path );
        std::vector<std::string> _ls( const std::string & path );

        //const SrvHandlers & handlers() const {return _handlers;}

    private:
        std::mutex _busy;

        CmdlArgs & _args;
        httplib::Server & _srv;
        LLRestUuid & _uuidg;
        Inference & _infr;

        SrvHandlers _handlers;
};

} // namespace LlamaREST

#endif