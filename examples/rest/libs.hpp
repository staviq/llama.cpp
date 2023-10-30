#ifndef LLAMA_CPP_REST_LIBS_HPP
#define LLAMA_CPP_REST_LIBS_HPP

#ifdef REST_WITH_ENCRYPTION
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif

#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH ((std::numeric_limits<size_t>::max)())

#include "lib/httplib.h"
#include "lib/json.hpp"
#include "lib/lz4.h"

using namespace httplib;
using json = nlohmann::json;

#endif