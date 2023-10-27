#ifndef LLAMA_CPP_REST_CMDLARGS_HPP
#define LLAMA_CPP_REST_CMDLARGS_HPP

#include "libs.hpp"

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <map>

namespace LlamaREST
{

using json = nlohmann::json;

class CmdlArgs{
    public:
        CmdlArgs( int argc, char ** argv )
        :
            _self(argc ? argv[0] : ""),
            _args(argv + (argc ? 1 : 0), argv + argc)
        {}
        ~CmdlArgs(){}
    private:
        CmdlArgs(CmdlArgs & other) = delete;
        void operator=(const CmdlArgs &)  = delete;

    public:
        enum class ReqDef{
            Required,
            Optional,
            Never
        };
        enum class TypDef{
            Any,
            UInteger,
            Integer,
            Float,
            Text,
            Ratio,
            Path,
            Directory,
            File
        };
        struct ValDef{
            ValDef(){}
            ValDef(
                ReqDef required,
                TypDef type = TypDef::Any,
                std::string placeholder = ""
            )
            :
                required(required),
                type(type),
                placeholder(placeholder)
            {}
            ReqDef required;
            TypDef type;
            std::string placeholder;
        };
        struct ArgDef{
            ArgDef(){}
            ArgDef(
                std::string name,
                std::set<std::string> aliases,
                std::string help,
                ReqDef required = ReqDef::Never,
                ValDef value = ValDef(ReqDef::Never)
            )
            :
                name(name),
                aliases(aliases),
                help(help),
                required(required),
                value(value)
            {}
            std::string name;
            std::set<std::string> aliases;
            std::string help;
            ReqDef required;
            ValDef value;
            bool operator==(const ArgDef& t) const
            {
                return (this->aliases == t.aliases);
            }
            size_t operator()(const ArgDef& p) const
            {
                size_t h = 0;
                for(const auto & alias: p.aliases)
                {
                    h ^= std::hash<std::string>()(alias);
                }
                return h;
            }
        };

    public:
        void arg (ArgDef def){
            const auto & newdef = *(_argdefs.insert(def).first);

            for (const auto & alias: newdef.aliases)
            {
                _aliases.emplace( alias, newdef );
            
            }
        }

        json parse(){
            for(const auto & def: _argdefs){
                if(def.required == ReqDef::Required){
                    bool found{false};
                    for(const auto & alias: def.aliases){
                        if(std::find(_args.begin(), _args.end(), alias) != _args.end()){
                            found = true;
                            break;
                        }
                    }
                    if(!found){
                        std::stringstream e;
                        e << "parameter "
                          << json( def.aliases ).dump()
                          << " is required:"
                          << std::endl
                          << "Parameter description:"
                          << std::endl
                          << def.help;
                        throw std::invalid_argument( e.str() );
                    }
                }
            }
            json j;
            j["self"] = _self;
            for(auto arg = _args.begin(); arg != _args.end(); std::advance(arg,1)){
                auto alias = _aliases.find((*arg));
                if(alias == _aliases.end()){
                    std::stringstream e;
                    e << "unrecognised parameter '"
                      << *arg
                      << "'";
                    throw std::invalid_argument( e.str() );
                }
                const auto & name = (*alias).first;
                if ( alias != _aliases.end()){
                    const auto & def = alias->second;
                    const auto & needs_value = def.value.required;
                    if (needs_value == ReqDef::Never){
                        j[def.name] = true;
                        continue;
                    }
                    if (needs_value == ReqDef::Optional || needs_value == ReqDef::Required){
                        if (
                            arg == _args.end()
                            || std::next(arg,1) == _args.end()
                            ||  _aliases.find(*std::next(arg,1)) != _aliases.end()
                        ){
                            if (needs_value == ReqDef::Required){
                                std::stringstream e;
                                e << "parameter '"
                                  << name
                                  << "' expects an argument: "
                                  << std::endl
                                  << "Parameter description:"
                                  << std::endl
                                  << def.help;
                                throw std::invalid_argument( e.str() );
                            }

                            j[def.name] = true;
                            continue;
                        }

                        switch(def.value.type){
                            case TypDef::Any:
                            case TypDef::Text: {
                                j[def.name] = *std::next(arg,1);
                                break;
                            }
                            case TypDef::Float: {
                                try{
                                    auto tmp = std::stold(*std::next(arg,1));
                                    j[def.name] = tmp;
                                } catch ( std::exception & ce ) {
                                    std::stringstream e;
                                    e << "parameter '"
                                      << name
                                      << "' expects FLOAT argument, got '"
                                      << *std::next(arg,1)
                                      << "': "
                                      << std::endl
                                      << "Parameter description:"
                                      << std::endl
                                      << def.help;
                                    throw std::invalid_argument( e.str() );
                                }
                                break;
                            }
                            case TypDef::Integer: {
                                auto tmp = std::stoll(*std::next(arg,1));
                                j[def.name] = tmp;
                                break;
                            }
                            case TypDef::UInteger: {
                                try{
                                    auto tmp = std::stoull(*std::next(arg,1));
                                    j[def.name] = tmp;
                                } catch ( std::exception & ce ) {
                                    std::stringstream e;
                                    e << "parameter '"
                                      << name
                                      << "' expects UNSIGNED_INTEGER argument, got '"
                                      << *std::next(arg,1)
                                      << "': "
                                      << std::endl
                                      << "Parameter description:"
                                      << std::endl
                                      << def.help;
                                    throw std::invalid_argument( e.str() );
                                }
                                break;
                            }
                            default: {
                                j[def.name] = *std::next(arg,1);
                                break;
                            }
                        }
                        std::advance(arg,1);
                        continue;
                    }
                }
            }
            _parsed = j;
            return _parsed;
        }

        json params() const {
            return _parsed;
        }

        std::string self() const {
            return _self;
        }
    private:
        std::string _self;
        std::vector<std::string> _args;
        std::unordered_set<ArgDef,ArgDef> _argdefs;
        std::map<std::string, const ArgDef &> _aliases;
        json _parsed;
};

} // namespace LlamaREST

#endif