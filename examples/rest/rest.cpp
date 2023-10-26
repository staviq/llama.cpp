#include "libs.hpp"

#include "build-info.h"
#include "grammar-parser.h"

#include "utils.hpp"
#include "system.hpp"
#include "cmdlargs.hpp"
#include "inference.hpp"

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

    json params;

    try{
        params = cmdlargs.parse();
    } catch ( std::exception & e){
        fprintf(stderr, "Error: %s\n", e.what());
        exit(1);
    }

    fprintf(stderr, "CMDL: %s\n", params.dump(1, '\t').c_str());

    LLRestUuid uuid_generator;
    json inference_config;
    Inference llama(inference_config);

#ifdef REST_WITH_ENCRYPTION
    httplib::SSLServer srv;
#else
    httplib::Server srv;
#endif

    System system( cmdlargs, srv, uuid_generator );

    auto endpoint_root = [](const Request &req, Response &res) {
        LLREST_PRINT_HEADERS(req);
        /*
        if( !req.has_header("X-LlamaCpp-Client") || req.get_header_value("X-LlamaCpp-Client").empty())
        {
            #include "help/root.html.hpp"
            res.set_content( std::string((const char *)root_html, root_html_len), "text/html");
        }
        else
        {
            res.set_content("Hello World! Client\n", "text/plain");
        }
        */
        json response;
        response["comment"] = "No endpoint specified. Try '/session'.";
        res.set_content( response.dump(), "application/json");
    };
    srv.Get ("/", endpoint_root);

    auto endpoint_session = [&](const Request &req, Response &res) {
        json response;
        response["session"] = uuid_generator.make();
        res.set_content( response.dump(), "application/json");
    };
    srv.Get("/session", endpoint_session);

    auto endpoint_auth = [](const Request &req, Response &res) {
        res.set_content("Auth", "text/plain");
    };
    srv.Get("/auth", endpoint_auth);

    auto endpoint_system = [](const Request &req, Response &res) {
        res.set_content("System", "text/plain");
    };
    srv.Get("/system", endpoint_auth);

    run = true;
    std::thread srv_t{[&] {
        srv.listen("0.0.0.0", 8080);
    }};
    if( srv_t.joinable() ) srv_t.join();

    return 0;
}