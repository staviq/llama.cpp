#ifndef LLAMA_CPP_REST_Chat_HPP
#define LLAMA_CPP_REST_Chat_HPP

#include "libs.hpp"
#include "utils.hpp"
#include "cmdlargs.hpp"
#include "inference.hpp"

#include "common.h"
#include "llama.h"

namespace LlamaREST
{

class Chat{
    public:
        Chat(CmdlArgs & cmdlargs, httplib::Server & srv, LLRestUuid & uuid_generator, Inference & infr);

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
