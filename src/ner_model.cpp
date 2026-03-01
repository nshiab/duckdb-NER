#include "ner_model.hpp"

#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

struct ner_hparams {
	int32_t n_vocab = 30522;
	int32_t n_max_tokens = 512;
	int32_t n_embd = 256;
	int32_t n_intermediate = 1536;
	int32_t n_head = 12;
	int32_t n_layer = 6;
	int32_t n_labels = 9; // Added for NER
	int32_t f16 = 1;
};

struct ner_layer {
	struct ggml_tensor *ln_att_w;
	struct ggml_tensor *ln_att_b;
	struct ggml_tensor *ln_out_w;
	struct ggml_tensor *ln_out_b;
	struct ggml_tensor *q_w;
	struct ggml_tensor *q_b;
	struct ggml_tensor *k_w;
	struct ggml_tensor *k_b;
	struct ggml_tensor *v_w;
	struct ggml_tensor *v_b;
	struct ggml_tensor *o_w;
	struct ggml_tensor *o_b;
	struct ggml_tensor *ff_i_w;
	struct ggml_tensor *ff_i_b;
	struct ggml_tensor *ff_o_w;
	struct ggml_tensor *ff_o_b;
};

struct ner_vocab {
	std::map<std::string, ner_vocab_id> token_to_id;
	std::map<std::string, ner_vocab_id> subword_token_to_id;
	std::map<ner_vocab_id, std::string> _id_to_token;
	std::map<ner_vocab_id, std::string> _id_to_subword_token;
};

struct ner_model {
	ner_hparams hparams;
	struct ggml_tensor *word_embeddings;
	struct ggml_tensor *token_type_embeddings;
	struct ggml_tensor *position_embeddings;
	struct ggml_tensor *ln_e_w;
	struct ggml_tensor *ln_e_b;
	std::vector<ner_layer> layers;
	// NER specific
	struct ggml_tensor *classifier_weight;
	struct ggml_tensor *classifier_bias;
	struct ggml_context *ctx;
	std::map<std::string, struct ggml_tensor *> tensors;
};

struct ner_buffer {
	uint8_t *data = NULL;
	size_t size = 0;
	void resize(size_t size) {
		delete[] data;
		data = new uint8_t[size];
		this->size = size;
	}
	~ner_buffer() {
		delete[] data;
	}
};

struct ner_ctx {
	ner_model model;
	ner_vocab vocab;
	size_t mem_per_token;
	int64_t mem_per_input;
	ner_buffer buf_compute;
};

// Simplified tokenizer based on bert.cpp
void ner_tokenize(struct ner_ctx *ctx, const char *text, ner_vocab_id *tokens, int32_t *n_tokens, int32_t n_max_tokens) {
	const auto &vocab = ctx->vocab;
	const int cls_tok_id = vocab.token_to_id.at("[CLS]");
	const int sep_tok_id = vocab.token_to_id.at("[SEP]");

	std::string str = text;
	std::vector<std::string> words;
	// Simple space-based split for now, WordPiece will handle the rest
	std::string word;
	for (char c : str) {
		if (isspace(c)) {
			if (!word.empty()) {
				words.push_back(word);
			}
			word.clear();
		} else {
			word += c;
		}
	}
	if (!word.empty()) {
		words.push_back(word);
	}

	int32_t t = 0;
	tokens[t++] = cls_tok_id;

	for (const auto &word : words) {
		if (t >= n_max_tokens - 1) {
			break;
		}

		int i = 0;
		int n = word.size();
		auto *token_map = &vocab.token_to_id;
		while (i < n) {
			if (t >= n_max_tokens - 1) {
				break;
			}
			int j = n;
			bool found = false;
			while (j > i) {
				auto it = token_map->find(word.substr(i, j - i));
				if (it != token_map->end()) {
					tokens[t++] = it->second;
					i = j;
					token_map = &vocab.subword_token_to_id;
					found = true;
					break;
				}
				--j;
			}
			if (!found) {
				token_map = &vocab.subword_token_to_id;
				++i; // skip unknown
			}
		}
	}
	tokens[t++] = sep_tok_id;
	*n_tokens = t;
}

struct ner_ctx *ner_load_from_file(const char *fname) {
	auto fin = std::ifstream(fname, std::ios::binary);
	if (!fin) {
		return nullptr;
	}

	uint32_t magic;
	fin.read((char *)&magic, sizeof(magic));
	if (magic != 0x67676d6c) {
		return nullptr;
	}

	ner_ctx *new_ner = new ner_ctx;
	auto &hparams = new_ner->model.hparams;
	fin.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
	fin.read((char *)&hparams.n_max_tokens, sizeof(hparams.n_max_tokens));
	fin.read((char *)&hparams.n_embd, sizeof(hparams.n_embd));
	fin.read((char *)&hparams.n_intermediate, sizeof(hparams.n_intermediate));
	fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
	fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
	fin.read((char *)&hparams.f16, sizeof(hparams.f16));
	// Check if there is space for n_labels in the header (we'll need our converter to write it)
	fin.read((char *)&hparams.n_labels, sizeof(hparams.n_labels));

	for (int i = 0; i < hparams.n_vocab; i++) {
		uint32_t len;
		fin.read((char *)&len, sizeof(len));
		std::string word(len, 0);
		fin.read((char *)word.data(), len);
		if (word.size() > 2 && word[0] == '#' && word[1] == '#') {
			new_ner->vocab.subword_token_to_id[word.substr(2)] = i;
			new_ner->vocab._id_to_subword_token[i] = word;
		} else {
			new_ner->vocab.token_to_id[word] = i;
			new_ner->vocab._id_to_token[i] = word;
		}
	}

	ggml_type wtype = (hparams.f16 == 2) ? GGML_TYPE_Q4_0 : (hparams.f16 == 1 ? GGML_TYPE_F16 : GGML_TYPE_F32);
	size_t ctx_size = 512 * 1024 * 1024; // 512MB for tensors, should be enough for BERT-base

	struct ggml_init_params params = {.mem_size = ctx_size, .mem_buffer = NULL, .no_alloc = false};
	new_ner->model.ctx = ggml_init(params);
	auto &model = new_ner->model;
	auto *ctx = model.ctx;

	model.word_embeddings = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, hparams.n_vocab);
	model.token_type_embeddings = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, 2);
	model.position_embeddings = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, hparams.n_max_tokens);
	model.ln_e_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
	model.ln_e_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
	model.classifier_weight = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, hparams.n_labels);
	model.classifier_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_labels);

	model.tensors["embeddings.word_embeddings.weight"] = model.word_embeddings;
	model.tensors["embeddings.token_type_embeddings.weight"] = model.token_type_embeddings;
	model.tensors["embeddings.position_embeddings.weight"] = model.position_embeddings;
	model.tensors["embeddings.LayerNorm.weight"] = model.ln_e_w;
	model.tensors["embeddings.LayerNorm.bias"] = model.ln_e_b;
	model.tensors["classifier.weight"] = model.classifier_weight;
	model.tensors["classifier.bias"] = model.classifier_bias;

	model.layers.resize(hparams.n_layer);
	for (int i = 0; i < hparams.n_layer; i++) {
		auto &layer = model.layers[i];
		layer.q_w = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, hparams.n_embd);
		layer.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
		layer.k_w = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, hparams.n_embd);
		layer.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
		layer.v_w = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, hparams.n_embd);
		layer.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
		layer.o_w = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, hparams.n_embd);
		layer.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
		layer.ln_att_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
		layer.ln_att_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
		layer.ff_i_w = ggml_new_tensor_2d(ctx, wtype, hparams.n_embd, hparams.n_intermediate);
		layer.ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_intermediate);
		layer.ff_o_w = ggml_new_tensor_2d(ctx, wtype, hparams.n_intermediate, hparams.n_embd);
		layer.ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
		layer.ln_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
		layer.ln_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);

		std::string base = "encoder.layer." + std::to_string(i) + ".";
		model.tensors[base + "attention.self.query.weight"] = layer.q_w;
		model.tensors[base + "attention.self.query.bias"] = layer.q_b;
		model.tensors[base + "attention.self.key.weight"] = layer.k_w;
		model.tensors[base + "attention.self.key.bias"] = layer.k_b;
		model.tensors[base + "attention.self.value.weight"] = layer.v_w;
		model.tensors[base + "attention.self.value.bias"] = layer.v_b;
		model.tensors[base + "attention.output.dense.weight"] = layer.o_w;
		model.tensors[base + "attention.output.dense.bias"] = layer.o_b;
		model.tensors[base + "attention.output.LayerNorm.weight"] = layer.ln_att_w;
		model.tensors[base + "attention.output.LayerNorm.bias"] = layer.ln_att_b;
		model.tensors[base + "intermediate.dense.weight"] = layer.ff_i_w;
		model.tensors[base + "intermediate.dense.bias"] = layer.ff_i_b;
		model.tensors[base + "output.dense.weight"] = layer.ff_o_w;
		model.tensors[base + "output.dense.bias"] = layer.ff_o_b;
		model.tensors[base + "output.LayerNorm.weight"] = layer.ln_out_w;
		model.tensors[base + "output.LayerNorm.bias"] = layer.ln_out_b;
	}

	while (true) {
		int32_t n_dims, length, ftype_in;
		fin.read((char *)&n_dims, sizeof(n_dims));
		fin.read((char *)&length, sizeof(length));
		fin.read((char *)&ftype_in, sizeof(ftype_in));
		if (fin.eof()) {
			break;
		}

		int64_t ne[2] = {1, 1};
		for (int i = 0; i < n_dims; i++) {
			int32_t ne_cur;
			fin.read((char *)&ne_cur, sizeof(ne_cur));
			ne[i] = ne_cur;
		}
		std::string name(length, 0);
		fin.read(&name[0], length);

		auto it = model.tensors.find(name);
		if (it == model.tensors.end()) {
			// Skip unknown tensor
			ggml_type t = (ftype_in == 0) ? GGML_TYPE_F32 : (ftype_in == 1 ? GGML_TYPE_F16 : GGML_TYPE_Q4_0);
			size_t row_size = (ggml_type_size(t) * ne[0]) / ggml_blck_size(t);
			fin.seekg(row_size * ne[1], std::ios::cur);
			continue;
		}
		auto *tensor = it->second;
		fin.read((char *)tensor->data, ggml_nbytes(tensor));
	}
	fin.close();

	new_ner->buf_compute.resize(128 * 1024 * 1024); // 128MB compute buffer
	new_ner->mem_per_token = 1024 * 1024;          // Dummy estimate
	return new_ner;
}

void ner_free(struct ner_ctx *ctx) {
	if (ctx) {
		if (ctx->model.ctx) {
			ggml_free(ctx->model.ctx);
		}
		delete ctx;
	}
}

void ner_eval(struct ner_ctx *ctx, int32_t n_threads, ner_vocab_id *tokens, int32_t n_tokens, float *logits) {
	const auto &model = ctx->model;
	const auto &hparams = model.hparams;

	const int n_embd = hparams.n_embd;
	const int n_layer = hparams.n_layer;
	const int n_head = hparams.n_head;
	const int d_head = n_embd / n_head;
	const int N = n_tokens;

	struct ggml_init_params params = {
	    .mem_size = ctx->buf_compute.size, .mem_buffer = ctx->buf_compute.data, .no_alloc = false};
	struct ggml_context *ctx0 = ggml_init(params);
	struct ggml_cgraph gf = {};

	struct ggml_tensor *token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
	memcpy(token_layer->data, tokens, N * sizeof(int32_t));

	struct ggml_tensor *token_types = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
	ggml_set_zero(token_types);

	struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
	for (int i = 0; i < N; i++) {
		ggml_set_i32_1d(positions, i, i);
	}

	struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.word_embeddings, token_layer);
	inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.token_type_embeddings, token_types), inpL);
	inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.position_embeddings, positions), inpL);

	// embd norm
	{
		inpL = ggml_norm(ctx0, inpL);
		inpL = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.ln_e_w, inpL), inpL),
		                ggml_repeat(ctx0, model.ln_e_b, inpL));
	}

	// layers
	for (int il = 0; il < n_layer; il++) {
		struct ggml_tensor *cur = inpL;

		// self-attention
		{
			struct ggml_tensor *Qcur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, cur),
			                                    ggml_mul_mat(ctx0, model.layers[il].q_w, cur));
			struct ggml_tensor *Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, d_head, n_head, N), 0, 2, 1, 3);

			struct ggml_tensor *Kcur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, cur),
			                                    ggml_mul_mat(ctx0, model.layers[il].k_w, cur));
			struct ggml_tensor *K = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, d_head, n_head, N), 0, 2, 1, 3);

			struct ggml_tensor *Vcur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, cur),
			                                    ggml_mul_mat(ctx0, model.layers[il].v_w, cur));
			struct ggml_tensor *V = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Vcur, d_head, n_head, N), 0, 2, 1, 3);

			struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
			KQ = ggml_soft_max(ctx0, ggml_scale(ctx0, KQ, ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head))));

			V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
			struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ);
			KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

			cur = ggml_cpy(ctx0, KQV, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
		}

		// attention output
		cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].o_b, cur), ggml_mul_mat(ctx0, model.layers[il].o_w, cur));
		cur = ggml_add(ctx0, cur, inpL);

		// attention norm
		{
			cur = ggml_norm(ctx0, cur);
			cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].ln_att_w, cur), cur),
			                ggml_repeat(ctx0, model.layers[il].ln_att_b, cur));
		}

		struct ggml_tensor *att_output = cur;

		// intermediate
		cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
		cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ff_i_b, cur), cur);
		cur = ggml_gelu(ctx0, cur);

		// output
		cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
		cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ff_o_b, cur), cur);
		cur = ggml_add(ctx0, att_output, cur);

		// output norm
		{
			cur = ggml_norm(ctx0, cur);
			cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].ln_out_w, cur), cur),
			                ggml_repeat(ctx0, model.layers[il].ln_out_b, cur));
		}

		inpL = cur;
	}

	// Classifier head: logits = inpL * classifier_weight + classifier_bias
	// inpL is [n_embd, N], weight is [n_embd, n_labels]
	// res will be [n_labels, N]
	struct ggml_tensor *res =
	    ggml_add(ctx0, ggml_mul_mat(ctx0, model.classifier_weight, inpL), ggml_repeat(ctx0, model.classifier_bias, inpL));

	ggml_build_forward_expand(&gf, res);
	ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

	memcpy(logits, ggml_get_data(res), N * hparams.n_labels * sizeof(float));
	ggml_free(ctx0);
}

int32_t ner_n_embd(struct ner_ctx *ctx) {
	return ctx->model.hparams.n_embd;
}
int32_t ner_n_max_tokens(struct ner_ctx *ctx) {
	return ctx->model.hparams.n_max_tokens;
}
int32_t ner_n_labels(struct ner_ctx *ctx) {
	return ctx->model.hparams.n_labels;
}
const char *ner_vocab_id_to_token(struct ner_ctx *ctx, ner_vocab_id id) {
	auto it = ctx->vocab._id_to_token.find(id);
	if (it != ctx->vocab._id_to_token.end()) {
		return it->second.c_str();
	}
	it = ctx->vocab._id_to_subword_token.find(id);
	if (it != ctx->vocab._id_to_subword_token.end()) {
		return it->second.c_str();
	}
	return "[UNK]";
}
