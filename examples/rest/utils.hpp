#ifndef LLAMA_CPP_REST_UTILS_HPP
#define LLAMA_CPP_REST_UTILS_HPP

#include "libs.hpp"

#include <random>
#include <string>
#include <sstream>
#include <iomanip>

namespace LlamaREST
{

#define LLREST_PRINT_REQUEST(req)                                               \
do{                                                                             \
    fprintf( stderr, "R: '%s':'%s'\n", req.method.c_str(), req.path.c_str() );     \
    for(const auto & h: req.headers)                                            \
    {                                                                           \
        fprintf( stderr, "H: '%s':'%s'\n", h.first.c_str(), h.second.c_str() ); \
    }                                                                           \
}while(false)

struct SrvHandlers{
    SrvHandlers(){}
    std::map<std::string,httplib::Server::Handler> get;
    std::map<std::string,httplib::Server::Handler> post;
    std::map<std::string,httplib::Server::Handler> put;
    std::map<std::string,httplib::Server::Handler> patch;
    std::map<std::string,httplib::Server::Handler> del;
    std::map<std::string,httplib::Server::Handler> options;
};

static void srv_register_handlers(httplib::Server & srv, const SrvHandlers & handlers )
{
    for (const auto & h: handlers.get) {srv.Get(h.first,h.second);}
    for (const auto & h: handlers.post) {srv.Post(h.first,h.second);}
    for (const auto & h: handlers.put) {srv.Put(h.first,h.second);}
    for (const auto & h: handlers.patch) {srv.Patch(h.first,h.second);}
    for (const auto & h: handlers.del) {srv.Delete(h.first,h.second);}
    for (const auto & h: handlers.options) {srv.Options(h.first,h.second);}
}
class JsonFile{
    public:
        JsonFile( std::string path, json content = json() )
        :
            _path(path),
            _content(content)
        {
            load();
        }
    public:
        json & content()
        {
            return _content;
        }

        json & load() { return load(_path);}
        json & load( std::string path )
        {
            std::ifstream f(path, std::ios::in);
            if(!f.is_open())
            {
                return _content;
            }
            std::stringstream buffer;
            buffer << f.rdbuf();
            _content = json::parse(buffer);
            return _content;
        }

        void save(){return save(_path);}
        void save( std::string path )
        {
            std::ofstream f(_path, std::ios::out | std::ios::trunc);
            f << _content.dump(4, ' ');
            _path = path;
        }

    private:
        std::string & _path;
        json _content;

};

class LLRestUuid
{
    public:
        LLRestUuid()
        {
            seed();
        }

    public:
        std::string make()
        {
            std::stringstream u;
            uint8_t buf[16];
            for( auto & b : buf)
            {
                b = rng();
            }
            buf[8] &= ~(1 << 6);
            buf[8] |= (1 << 7);
            buf[6] &= ~(1 << 4);
            buf[6] &= ~(1 << 5);
            buf[6] |= (1 << 6);
            buf[6] &= ~(1 << 7);
            
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[0];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[1];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[2];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[3];
            u << "-";
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[4];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[5];
            u << "-";
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[6];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[7];
            u << "-";
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[8];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[9];
            u << "-";
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[10];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[11];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[12];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[13];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[14];
            u << std::nouppercase << std::setfill('0') << std::setw(2) << std::hex << (int)buf[15];

            return u.str();
        }
        
        operator std::string()
        {
            return make();
        }


    private:
        void seed()
        {
            rng.seed( (uint_fast32_t) this );
        }

    private:
        std::mt19937 rng;
};

} // namespace LlamaREST

#endif