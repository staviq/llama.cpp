#pragma once

#include <random>
#include <string>
#include <sstream>
#include <iomanip>

#define LLREST_PRINT_HEADERS(req)                                               \
    do{                                                                         \
    for(const auto & h: req.headers)                                            \
    {                                                                           \
        fprintf( stderr, "H: '%s':'%s'\n", h.first.c_str(), h.second.c_str() ); \
    }                                                                           \
    }while(false)

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
            return std::move(make());
        }


    private:
        void seed()
        {
            rng.seed( (uint_fast32_t) this );
        }

    private:
        std::mt19937 rng;
};
