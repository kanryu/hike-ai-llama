#include "llama.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

struct hike_llama_context {
    struct llama_context * native;
    char * system_prompt;
};

static enum ggml_log_level hike_llama_log_level = GGML_LOG_LEVEL_WARN;

static void hike_llama_log_callback(enum ggml_log_level level,
                                    const char * text, void * user_data) {
    (void) user_data;
    if (level >= hike_llama_log_level && text != NULL) {
        fputs(text, stderr);
    }
}

static char * hike_llama_copy_string(const char * value) {
    size_t length;
    char * copy;
    if (value == NULL) return NULL;
    length = strlen(value);
    copy = (char *) malloc(length + 1);
    if (copy == NULL) return NULL;
    memcpy(copy, value, length + 1);
    return copy;
}

static int32_t hike_llama_strip_thinking(uint8_t * output, int32_t length) {
    static const char * begin = "<think>";
    static const char * end = "</think>";
    int32_t read = 0;
    int32_t write = 0;
    int in_thinking = 0;

    while (read < length) {
        const char * marker = in_thinking ? end : begin;
        int32_t marker_length = (int32_t) strlen(marker);
        if (read + marker_length <= length &&
            memcmp(output + read, marker, (size_t) marker_length) == 0) {
            read += marker_length;
            in_thinking = !in_thinking;
            continue;
        }
        if (!in_thinking) output[write++] = output[read];
        ++read;
    }
    return write;
}

void hike_llama_set_console_utf8(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

void hike_llama_set_log_level(int32_t level) {
    if (level < GGML_LOG_LEVEL_NONE) level = GGML_LOG_LEVEL_NONE;
    if (level > GGML_LOG_LEVEL_ERROR) level = GGML_LOG_LEVEL_ERROR;
    hike_llama_log_level = (enum ggml_log_level) level;
    llama_log_set(hike_llama_log_callback, NULL);
}

static char * hike_llama_format_prompt(struct hike_llama_context * context,
                                       const char * prompt) {
    struct llama_chat_message messages[2];
    const char * tmpl;
    char * formatted;
    size_t message_count = 1;
    size_t total_length = strlen(prompt);
    int32_t capacity;
    int32_t length;

    messages[0].role = "user";
    messages[0].content = prompt;
    if (context->system_prompt != NULL && context->system_prompt[0] != '\0') {
        messages[0].role = "system";
        messages[0].content = context->system_prompt;
        messages[1].role = "user";
        messages[1].content = prompt;
        message_count = 2;
        total_length += strlen(context->system_prompt);
    }

    tmpl = llama_model_chat_template(llama_get_model(context->native), NULL);
    if (tmpl == NULL) return hike_llama_copy_string(prompt);

    capacity = (int32_t) (total_length * 2 + 128);
    if (capacity < 128) capacity = 128;
    formatted = (char *) malloc((size_t) capacity);
    if (formatted == NULL) return NULL;
    length = llama_chat_apply_template(tmpl, messages, message_count, true,
                                       formatted, capacity);
    if (length < 0) {
        free(formatted);
        return NULL;
    }
    if (length >= capacity) {
        char * resized = (char *) realloc(formatted, (size_t) length + 1);
        if (resized == NULL) {
            free(formatted);
            return NULL;
        }
        formatted = resized;
        length = llama_chat_apply_template(tmpl, messages, message_count, true,
                                           formatted, length + 1);
        if (length < 0) {
            free(formatted);
            return NULL;
        }
    }
    formatted[length] = '\0';
    return formatted;
}

void hike_llama_backend_init(void) {
    llama_backend_init();
}

void hike_llama_backend_free(void) {
    llama_backend_free();
}

struct llama_model * hike_llama_load_model(const char * path,
                                            int32_t gpu_layers) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = gpu_layers;
    return llama_model_load_from_file(path, params);
}

void hike_llama_free_model(struct llama_model * model) {
    if (model != NULL) llama_model_free(model);
}

struct hike_llama_context * hike_llama_create_context(
    struct llama_model * model, int32_t context_size, int32_t batch_size,
    int32_t threads) {
    if (model == NULL) return NULL;
    struct llama_context_params params = llama_context_default_params();
    if (context_size > 0) params.n_ctx = (uint32_t) context_size;
    if (batch_size > 0) params.n_batch = (uint32_t) batch_size;
    if (threads > 0) {
        params.n_threads = threads;
        params.n_threads_batch = threads;
    }
    struct llama_context * native = llama_init_from_model(model, params);
    struct hike_llama_context * context;
    if (native == NULL) return NULL;
    context = (struct hike_llama_context *) calloc(1, sizeof(*context));
    if (context == NULL) {
        llama_free(native);
        return NULL;
    }
    context->native = native;
    return context;
}

int32_t hike_llama_embedding_dimension(struct llama_model * model) {
    if (model == NULL) return -1;
    return llama_model_n_embd_out(model);
}

int32_t hike_llama_embed(struct hike_llama_context * context,
                         const char * text, float * output,
                         int32_t output_capacity) {
    const struct llama_vocab * vocab;
    int32_t text_len;
    int32_t token_capacity;
    int32_t token_count;
    llama_token * tokens;
    float * embedding;
    int32_t dimension;
    if (context == NULL || context->native == NULL || text == NULL ||
        output == NULL || output_capacity <= 0) return -1;
    vocab = llama_model_get_vocab(llama_get_model(context->native));
    text_len = (int32_t) strlen(text);
    token_capacity = text_len + 256;
    tokens = (llama_token *) malloc((size_t) token_capacity * sizeof(*tokens));
    if (tokens == NULL) return -1;
    token_count = llama_tokenize(vocab, text, text_len, tokens,
                                 token_capacity, true, false);
    if (token_count < 0) {
        token_capacity = -token_count;
        tokens = (llama_token *) realloc(tokens,
            (size_t) token_capacity * sizeof(*tokens));
        if (tokens == NULL) return -1;
        token_count = llama_tokenize(vocab, text, text_len, tokens,
                                     token_capacity, true, false);
    }
    if (token_count <= 0) { free(tokens); return -1; }
    if (llama_decode(context->native,
                     llama_batch_get_one(tokens, token_count)) != 0) {
        free(tokens);
        return -1;
    }
    embedding = llama_get_embeddings(context->native);
    dimension = llama_model_n_embd_out(llama_get_model(context->native));
    if (embedding == NULL || dimension <= 0 || output_capacity < dimension) {
        free(tokens);
        return -1;
    }
    memcpy(output, embedding, (size_t) dimension * sizeof(float));
    free(tokens);
    return dimension;
}

void hike_llama_free_context(struct hike_llama_context * context) {
    if (context == NULL) return;
    free(context->system_prompt);
    llama_free(context->native);
    free(context);
}

int32_t hike_llama_reset_context(struct hike_llama_context * context) {
    if (context == NULL || context->native == NULL) return -1;
    llama_memory_clear(llama_get_memory(context->native), true);
    return 0;
}

int32_t hike_llama_save_state(struct hike_llama_context * context,
                              const char * path) {
    FILE * file;
    uint8_t * data;
    size_t size;
    size_t written;
    if (context == NULL || context->native == NULL || path == NULL) return -1;
    size = llama_state_get_size(context->native);
    data = (uint8_t *) malloc(size);
    if (data == NULL) return -1;
    written = llama_state_get_data(context->native, data, size);
    file = fopen(path, "wb");
    if (file == NULL) { free(data); return -1; }
    if (fwrite(data, 1, written, file) != written) {
        fclose(file);
        free(data);
        return -1;
    }
    fclose(file);
    free(data);
    return 0;
}

int32_t hike_llama_load_state(struct hike_llama_context * context,
                              const char * path) {
    FILE * file;
    long length;
    uint8_t * data;
    size_t read_length;
    size_t restored;
    if (context == NULL || context->native == NULL || path == NULL) return -1;
    file = fopen(path, "rb");
    if (file == NULL) return -1;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return -1; }
    length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    data = (uint8_t *) malloc((size_t) length);
    if (data == NULL) { fclose(file); return -1; }
    read_length = fread(data, 1, (size_t) length, file);
    fclose(file);
    if (read_length != (size_t) length) { free(data); return -1; }
    restored = llama_state_set_data(context->native, data, read_length);
    free(data);
    return restored == read_length ? 0 : -1;
}

int32_t hike_llama_set_system_prompt(struct hike_llama_context * context,
                                      const char * prompt) {
    char * copy;
    if (context == NULL || prompt == NULL) return -1;
    copy = hike_llama_copy_string(prompt);
    if (copy == NULL) return -1;
    free(context->system_prompt);
    context->system_prompt = copy;
    return 0;
}

int32_t hike_llama_generate(struct hike_llama_context * context,
                            const char * prompt,
                            int32_t max_tokens,
                            float temperature,
                            float top_p,
                            uint32_t seed,
                            uint8_t * output,
                            int32_t output_capacity) {
    if (context == NULL || context->native == NULL || prompt == NULL ||
        max_tokens <= 0 ||
        output == NULL || output_capacity <= 0) return -1;

    char * formatted_prompt = hike_llama_format_prompt(context, prompt);
    if (formatted_prompt == NULL) return -1;
    const struct llama_model * model = llama_get_model(context->native);
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t prompt_len = (int32_t) strlen(formatted_prompt);
    int32_t token_capacity = prompt_len + 256;
    llama_token * tokens = (llama_token *) malloc(
        (size_t) token_capacity * sizeof(llama_token));
    if (tokens == NULL) {
        free(formatted_prompt);
        return -1;
    }

    int32_t token_count = llama_tokenize(vocab, formatted_prompt, prompt_len, tokens,
                                         token_capacity, true, true);
    if (token_count < 0) {
        token_capacity = -token_count;
        tokens = (llama_token *) realloc(tokens,
            (size_t) token_capacity * sizeof(llama_token));
        if (tokens == NULL) {
            free(formatted_prompt);
            return -1;
        }
        token_count = llama_tokenize(vocab, formatted_prompt, prompt_len, tokens,
                                     token_capacity, true, true);
    }
    if (token_count <= 0 || token_count > token_capacity) {
        free(tokens);
        free(formatted_prompt);
        return -1;
    }

    struct llama_batch prompt_batch = llama_batch_get_one(tokens, token_count);
    if (llama_decode(context->native, prompt_batch) != 0) {
        free(tokens);
        free(formatted_prompt);
        return -1;
    }

    struct llama_sampler_chain_params sampler_params =
        llama_sampler_chain_default_params();
    struct llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (sampler == NULL) {
        free(tokens);
        free(formatted_prompt);
        return -1;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

    int32_t written = 0;
    llama_token last = 0;
    for (int32_t i = 0; i < max_tokens; ++i) {
        char piece[512];
        last = llama_sampler_sample(sampler, context->native, -1);
        if (llama_vocab_is_eog(vocab, last) ||
            llama_vocab_is_control(vocab, last)) break;
        int32_t piece_len = llama_token_to_piece(vocab, last, piece,
                                                 (int32_t) sizeof(piece),
                                                 0, false);
        if (piece_len < 0) break;
        int32_t remaining = output_capacity - written;
        int32_t copy_len = piece_len < remaining ? piece_len : remaining;
        if (copy_len > 0) {
            memcpy(output + written, piece, (size_t) copy_len);
            written += copy_len;
        }
        if (copy_len != piece_len) break;
        struct llama_batch next = llama_batch_get_one(&last, 1);
        if (llama_decode(context->native, next) != 0) break;
    }

    llama_sampler_free(sampler);
    free(tokens);
    free(formatted_prompt);
    return hike_llama_strip_thinking(output, written);
}
