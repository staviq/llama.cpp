#include "build-info.h"
#include "grammar-parser.h"

#include "libs.hpp"
#include "utils.hpp"
#include "system.hpp"
#include "cmdlargs.hpp"
#include "inference.hpp"
#include "chat.hpp"

#include "public.hpp"

#include <unistd.h>
#include <cstdio>

#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

using namespace LlamaREST;

std::atomic<bool> run   {false};

int main(int argc, char ** argv)
{
    CmdlArgs cmdlargs(argc,argv);

    cmdlargs.arg({
        "test",
        {"--test"},
        "Test parameter.",
        CmdlArgs::ReqDef::Optional,
        {CmdlArgs::ReqDef::Optional, CmdlArgs::TypDef::UInteger, "TEST"}
    });
    cmdlargs.arg({
        "verbose",
        {"-V", "--verbose"},
        "Enable detailed logging.",
        CmdlArgs::ReqDef::Optional,
        {CmdlArgs::ReqDef::Never}
    });
    cmdlargs.arg({
        "mdir",
        {"-md", "--models-directory"},
        "Models directory, used for managing models at runtime.",
        CmdlArgs::ReqDef::Optional,
        {CmdlArgs::ReqDef::Required, CmdlArgs::TypDef::Directory, "DIR"}
    });
    cmdlargs.arg({
        "model",
        {"-m", "--model"},
        "Model to be preloaded at startup.",
        CmdlArgs::ReqDef::Optional,
        {CmdlArgs::ReqDef::Required, CmdlArgs::TypDef::File, "MODEL"}
    });
    cmdlargs.arg({
        "srvhost",
        {"--host"},
        "IP address to listen on. Use 0.0.0.0 to listen on all interfaces. (default: 127.0.0.1)",
        CmdlArgs::ReqDef::Optional,
        {CmdlArgs::ReqDef::Required, CmdlArgs::TypDef::Text, "IP"}
    });
    cmdlargs.arg({
        "srvport",
        {"--port"},
        "Port to listen on. (default: 8080)",
        CmdlArgs::ReqDef::Optional,
        {CmdlArgs::ReqDef::Required, CmdlArgs::TypDef::UInteger, "PORT"}
    });
    cmdlargs.arg({
        "logdisable",
        {"--log-disable"},
        "Log disable",
        CmdlArgs::ReqDef::Optional,
        {CmdlArgs::ReqDef::Never, CmdlArgs::TypDef::Any, ""}
    });

    json params;

    try {
        params = cmdlargs.parse();
    } catch ( std::exception & e) {
        fprintf(stderr, "Error: %s\n", e.what());
        exit(1);
    }

    if (!params.contains("mdir")) {
        params["mdir"] = "./models";
    }
    if (!params.contains("srvhost")) {
        params["srvhost"] = "0.0.0.0";
    }
    if (!params.contains("srvport")) {
        params["srvport"] = 8081;
    }

    if (params.contains("logdisable") && params["logdisable"].get<bool>()) {
        log_disable();
    }

    fprintf(stderr, "CMDL: %s\n", params.dump(1, '\t').c_str());

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("rest", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    LLRestUuid uuid_generator;

#ifdef REST_WITH_ENCRYPTION
    httplib::SSLServer srv;
#else
    httplib::Server srv;
#endif

    srv.set_pre_routing_handler([](const Request & req, Response & res) {
        /*LLREST_PRINT_REQUEST(req);*/
        static std::set<std::string> dbg_headers = { "Content-Type", "Content-Length", "REMOTE_ADDR", "User-Agent"};
        fprintf( stderr, "R: %s\n\tM: '%s'\n", req.path.c_str(), req.method.c_str() );
        for(const auto & h: req.headers)
        {
            if (dbg_headers.find(h.first)!=dbg_headers.end()) {
                fprintf( stderr, "\tH: '%s':'%s'\n", h.first.c_str(), h.second.c_str() );
            }
        }
        return Server::HandlerResponse::Unhandled;
    });

    Inference inference( cmdlargs, srv, uuid_generator );
    System       system( cmdlargs, srv, uuid_generator, inference );
    Chat           chat( cmdlargs, srv, uuid_generator, inference );

    auto endpoint_index = [&](const Request &req, Response &res) {
        res.set_content( std::string((const char *)index_html, index_html_len), "text/html");
    };
    srv.Get("/", endpoint_index);
    srv.Get("/index.html", endpoint_index);

    auto endpoint_chat = [&](const Request &req, Response &res) {
        res.set_content( std::string((const char *)chat_html, chat_html_len), "text/html");
    };
    srv.Get("/chat", endpoint_chat);
    srv.Get("/chat.html", endpoint_chat);

    std::string basedir = "examples/rest/public";
    srv.set_base_dir(basedir);

    run = true;

    std::thread srv_t{[&] {
        if (!srv.listen(params["srvhost"], params["srvport"])) {
            LOG_TEE("Server failed to start.\n");
        }
    }};

    if( srv_t.joinable() ) {
        srv_t.join();
    }

    return 0;
}