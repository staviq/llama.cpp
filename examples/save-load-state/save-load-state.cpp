#include "common.h"
#include "llama.h"

#include "lz4.h"

#include <vector>
#include <cstdio>
#include <chrono>

#define C_ACCEL 1

static void run_screaming(const char* message, const int code) {
    fprintf(stderr, "%s (%d)\n", message, code);
    GGML_ASSERT(false);
}

int main(int argc, char ** argv) {
    gpt_params params;

    params.prompt = "The quick brown fox";

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    print_build_info();

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    auto n_past = 0;

    std::string result0;
    std::string result1;

    // init
    llama_model * model;
    llama_context * ctx;

    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // tokenize prompt
    auto tokens = llama_tokenize(ctx, params.prompt, true);

    // evaluate prompt
    llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), n_past, 0));
    n_past += tokens.size();

    // save state (rng, logits, embedding and kv_cache) to file
    {
        std::vector<uint8_t> state_mem(llama_get_state_size(ctx));
        FILE *fp_write = fopen("dump_state.bin.lz4", "wb");
        llama_copy_state_data(ctx, state_mem.data()); // could also copy directly to memory mapped file

        const char* const src = (const char*) state_mem.data();
        const int src_size = (int)(state_mem.size());
        const int max_dst_size = LZ4_compressBound(src_size);

        char* compressed_data = (char*)malloc((size_t)max_dst_size);
        if (compressed_data == NULL) {
            run_screaming("Failed to allocate memory for *compressed_data.", 1);
        }

        const int compressed_data_size = LZ4_compress_fast(src, compressed_data, src_size, max_dst_size, C_ACCEL);

        if (compressed_data_size <= 0) {
            run_screaming("A 0 or negative result from LZ4_compress_default() indicates a failure trying to compress the data. ", 1);
        }
        if (compressed_data_size > 0) {
            printf("\n#### We successfully compressed some data! Ratio: %.2f\n", (float) compressed_data_size/src_size);
        }
        
        compressed_data = (char *)realloc(compressed_data, (size_t)compressed_data_size);
        if (compressed_data == NULL) {
            run_screaming("Failed to re-alloc memory for compressed_data.  Sad :(", 1);
        }

        fwrite(compressed_data, 1, compressed_data_size, fp_write);
        fclose(fp_write);
    }

    // save state (last tokens)
    const auto n_past_saved = n_past;

    // first run
    printf("\nfirst run: %s", params.prompt.c_str());

    auto start = std::chrono::steady_clock::now();
    int tpscnt = 0;

    for (auto i = 0; i < params.n_predict; i++) {
        auto * logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(model);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        //auto next_token = llama_sample_token(ctx, &candidates_p);
        auto next_token = llama_sample_token_greedy(ctx, &candidates_p);
        auto next_token_str = llama_token_to_piece(ctx, next_token);

        tpscnt++;

        if(next_token_str[0] == '\n')
        {
            auto end = std::chrono::steady_clock::now();
            double timer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            auto tps = 1000.0/(timer/(double)tpscnt);

            tpscnt = 0;
            start = end;

            fprintf(stderr, "\n#### TPS: %0.2f\n", tps);
        }
        printf("%s", next_token_str.c_str());
        fflush(stdout);
        result0 += next_token_str;

        if (llama_decode(ctx, llama_batch_get_one(&next_token, 1, n_past, 0))) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_free(ctx);
            llama_free_model(model);
            return 1;
        }
        n_past += 1;
    }

    {
            auto end = std::chrono::steady_clock::now();
            double timer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            auto tps = 1000.0/(timer/(double)tpscnt);

            tpscnt = 0;
            start = end;

            fprintf(stderr, "\n#### END TPS: %0.2f\n", tps);
        }

    // save state (rng, logits, embedding and kv_cache) to file
    {
        std::vector<uint8_t> state_mem(llama_get_state_size(ctx));
        FILE *fp_write = fopen("dump_state2.bin.lz4", "wb");
        llama_copy_state_data(ctx, state_mem.data()); // could also copy directly to memory mapped file

        const char* const src = (const char*) state_mem.data();
        const int src_size = (int)(state_mem.size());
        const int max_dst_size = LZ4_compressBound(src_size);

        char* compressed_data = (char*)malloc((size_t)max_dst_size);
        if (compressed_data == NULL) {
            run_screaming("Failed to allocate memory for *compressed_data.", 1);
        }

        const int compressed_data_size = LZ4_compress_fast(src, compressed_data, src_size, max_dst_size, C_ACCEL);

        if (compressed_data_size <= 0) {
            run_screaming("A 0 or negative result from LZ4_compress_default() indicates a failure trying to compress the data. ", 1);
        }
        if (compressed_data_size > 0) {
            printf("\n#### We successfully compressed some data! Ratio: %.2f\n", (float) compressed_data_size/src_size);
        }
        
        compressed_data = (char *)realloc(compressed_data, (size_t)compressed_data_size);
        if (compressed_data == NULL) {
            run_screaming("Failed to re-alloc memory for compressed_data.  Sad :(", 1);
        }

        fwrite(compressed_data, 1, compressed_data_size, fp_write);
        fclose(fp_write);
    }

    printf("\n\n");

    // free old context
    llama_free(ctx);

    // make new context
    auto * ctx2 = llama_new_context_with_model(model, llama_context_params_from_gpt_params(params));

    printf("\nsecond run: %s", params.prompt.c_str());

    // load state (rng, logits, embedding and kv_cache) from file
    {
        std::vector<uint8_t> state_mem(llama_get_state_size(ctx2));
        FILE * fp_read = fopen("dump_state.bin.lz4", "rb");
        const size_t ret = fread(state_mem.data(), 1, state_mem.size(), fp_read);

        const int src_size = (int)(state_mem.size());

        const char* const compressed_data = (const char*) state_mem.data();
        const int compressed_data_size = ret;

        char* const regen_buffer = (char*)malloc(src_size);
        if (regen_buffer == NULL) {
            run_screaming("Failed to allocate memory for *regen_buffer.", 1);
        }
        const int decompressed_size = LZ4_decompress_safe(compressed_data, regen_buffer, compressed_data_size, src_size);
        if (decompressed_size < 0) {
            run_screaming("A negative result from LZ4_decompress_safe indicates a failure trying to decompress the data.  See exit code (echo $?) for value returned.", decompressed_size);
        }
        if (decompressed_size >= 0) {
            fprintf(stderr, "We successfully decompressed some data!\n");
        }
        if (decompressed_size != src_size) {
            run_screaming("Decompressed data is different from original! \n", 1);
        }

        memcpy(state_mem.data(), regen_buffer, decompressed_size);
        free(regen_buffer);
        
        if ((size_t)decompressed_size != state_mem.size()) {
            fprintf(stderr, "\n%s : failed to read state\n", __func__);
            llama_free(ctx2);
            llama_free_model(model);
            return 1;
        }

        llama_set_state_data(ctx2, state_mem.data());

        fclose(fp_read);
    }

    // restore state (last tokens)
    n_past = n_past_saved;

    // second run
    for (auto i = 0; i < params.n_predict; i++) {
        auto * logits = llama_get_logits(ctx2);
        auto n_vocab = llama_n_vocab(model);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        //auto next_token = llama_sample_token(ctx, &candidates_p);
        auto next_token = llama_sample_token_greedy(ctx, &candidates_p);
        auto next_token_str = llama_token_to_piece(ctx2, next_token);

        printf("%s", next_token_str.c_str());
        fflush(stdout);
        result1 += next_token_str;

        if (llama_decode(ctx2, llama_batch_get_one(&next_token, 1, n_past, 0))) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_free(ctx2);
            llama_free_model(model);
            return 1;
        }
        n_past += 1;
    }

    printf("\n");

    llama_free(ctx2);
    llama_free_model(model);

    if (result0 != result1) {
        fprintf(stderr, "\n%s : error : the 2 generations are different\n", __func__);
        return 1;
    }

    fprintf(stderr, "\n%s : success\n", __func__);

    return 0;
}
