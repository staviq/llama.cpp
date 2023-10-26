#include "libs.hpp"

#include "common.h"
#include "llama.h"

#include <mutex>

using json = nlohmann::json;

class Inference
{
    public:
        Inference( json config ){}
        ~Inference(){}
    private:
        Inference(Inference & other) = delete;
        void operator=(const Inference &)  = delete;

    private:
        std::mutex _busy;

        std::string _m_dir {"."};
};