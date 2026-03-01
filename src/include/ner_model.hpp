#ifndef NER_MODEL_HPP
#define NER_MODEL_HPP

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ner_ctx;

typedef int32_t ner_vocab_id;

struct ner_ctx *ner_load_from_file(const char *fname);
void ner_free(struct ner_ctx *ctx);

void ner_tokenize(struct ner_ctx *ctx, const char *text, ner_vocab_id *tokens, int32_t *n_tokens, int32_t n_max_tokens);

// Returns logits for each token: [n_tokens, n_labels]
void ner_eval(struct ner_ctx *ctx, int32_t n_threads, ner_vocab_id *tokens, int32_t n_tokens, float *logits);

int32_t ner_n_embd(struct ner_ctx *ctx);
int32_t ner_n_max_tokens(struct ner_ctx *ctx);
int32_t ner_n_labels(struct ner_ctx *ctx);

const char *ner_vocab_id_to_token(struct ner_ctx *ctx, ner_vocab_id id);

#ifdef __cplusplus
}
#endif

#endif // NER_MODEL_HPP
