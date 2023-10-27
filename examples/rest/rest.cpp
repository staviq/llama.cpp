#include "build-info.h"
#include "grammar-parser.h"

#include "libs.hpp"
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

    if (!params.contains("mdir")){
        params["mdir"] = "./models";
    }

    fprintf(stderr, "CMDL: %s\n", params.dump(1, '\t').c_str());

    LLRestUuid uuid_generator;
    Inference llama;

#ifdef REST_WITH_ENCRYPTION
    httplib::SSLServer srv;
#else
    httplib::Server srv;
#endif

    srv.set_pre_routing_handler([](const Request & req, Response & res) {
        LLREST_PRINT_REQUEST(req);
        return Server::HandlerResponse::Unhandled;
    });

    System system( cmdlargs, srv, uuid_generator, llama );

    auto endpoint_root = [](const Request &req, Response &res) {
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

    srv.Get("/session", [&](const Request &req, Response &res) {
        json response;
        response["session"] = uuid_generator.make();
        res.set_content( response.dump(), "application/json");
    });

    srv.Get("/auth", [](const Request &req, Response &res) {
        res.set_content("Auth", "text/plain");
    });

    run = true;
    std::thread srv_t{[&] {
        srv.listen("0.0.0.0", 8080);
    }};
    if( srv_t.joinable() ) srv_t.join();

    return 0;
}