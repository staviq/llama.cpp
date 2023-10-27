#ifndef LLAMA_CPP_REST_LIBS_HPP
#define LLAMA_CPP_REST_LIBS_HPP

#ifdef REST_WITH_ENCRYPTION
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif

#include "lib/httplib.h"
#include "lib/json.hpp"

using namespace httplib;
using json = nlohmann::json;

#endif