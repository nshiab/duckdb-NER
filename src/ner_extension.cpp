#define DUCKDB_EXTENSION_MAIN

#include "ner_extension.hpp"
#include "ner_model.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/config.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

struct NerGlobalState {
    struct ner_ctx * ctx = nullptr;
    std::string model_path;
};

static NerGlobalState global_state;

static void LoadModel(const std::string & path) {
    if (global_state.ctx) {
        ner_free(global_state.ctx);
        global_state.ctx = nullptr;
    }
    global_state.ctx = ner_load_from_file(path.c_str());
    global_state.model_path = path;
}

struct Entity {
    std::string text;
    std::string label;
};

inline void NerScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &input_vector = args.data[0];
    auto count = args.size();

    bool truncate_opt = true; // Default to true as per prompt "truncate=true instructs to silently truncate"
    if (args.ColumnCount() > 1) {
        UnifiedVectorFormat trunc_data;
        args.data[1].ToUnifiedFormat(count, trunc_data);
        auto trunc_vals = UnifiedVectorFormat::GetData<bool>(trunc_data);
        if (trunc_data.validity.RowIsValid(trunc_data.sel->get_index(0))) {
            truncate_opt = trunc_vals[trunc_data.sel->get_index(0)];
        }
    }

    if (!global_state.ctx) {
        ListVector::SetListSize(result, 0);
        result.SetVectorType(VectorType::FLAT_VECTOR);
        auto result_data = FlatVector::GetData<list_entry_t>(result);
        for (size_t i = 0; i < count; i++) {
            result_data[i] = {0, 0};
        }
        return;
    }

    auto &child_vector = ListVector::GetEntry(result);
    auto &entity_vector = StructVector::GetEntries(child_vector)[0];
    auto &label_vector = StructVector::GetEntries(child_vector)[1];

    list_entry_t *result_data = FlatVector::GetData<list_entry_t>(result);
    size_t current_offset = 0;

    UnifiedVectorFormat input_data;
    input_vector.ToUnifiedFormat(count, input_data);
    auto inputs = UnifiedVectorFormat::GetData<string_t>(input_data);

    int n_labels = ner_n_labels(global_state.ctx);
    int n_max_tokens = ner_n_max_tokens(global_state.ctx);
    
    std::vector<float> logits;
    std::vector<ner_vocab_id> tokens;
    tokens.resize(n_max_tokens);
    logits.resize(n_max_tokens * n_labels);

    const char * label_map[] = {"O", "MISC", "MISC", "PER", "PER", "ORG", "ORG", "LOC", "LOC"};

    for (size_t i = 0; i < count; i++) {
        auto idx = input_data.sel->get_index(i);
        if (!input_data.validity.RowIsValid(idx)) {
            FlatVector::SetNull(result, i, true);
            continue;
        }

        auto input_str = inputs[idx].GetString();
        result_data[i].offset = current_offset;
        
        int32_t n_tokens = 0;
        ner_tokenize(global_state.ctx, input_str.c_str(), tokens.data(), &n_tokens, n_max_tokens);
        
        // Simple heuristic: if we used exactly n_max_tokens, it might have been truncated.
        // In a real tokenizer we'd know for sure.
        if (!truncate_opt && n_tokens >= n_max_tokens) {
             throw InvalidInputException("Input string exceeds model token limit and truncate=false");
        }
        
        ner_eval(global_state.ctx, 4, tokens.data(), n_tokens, logits.data());

        std::vector<Entity> entities;
        Entity current_entity;
        int last_label_type = 0; // 0: O, 1: MISC, 2: PER, 3: ORG, 4: LOC

        for (int t = 0; t < n_tokens; t++) {
            int best_label = 0;
            float max_logit = -1e10;
            for (int l = 0; l < n_labels; l++) {
                if (logits[t * n_labels + l] > max_logit) {
                    max_logit = logits[t * n_labels + l];
                    best_label = l;
                }
            }

            std::string token_str = ner_vocab_id_to_token(global_state.ctx, tokens[t]);
            if (token_str == "[CLS]" || token_str == "[SEP]") continue;

            bool is_subword = (token_str.size() > 2 && token_str[0] == '#' && token_str[1] == '#');
            std::string clean_token = is_subword ? token_str.substr(2) : token_str;

            int label_type = (best_label + 1) / 2; // Maps B-X and I-X to same group
            if (best_label == 0) label_type = 0;

            if (label_type != 0) {
                if (label_type == last_label_type && (best_label % 2 == 0 || is_subword)) {
                    current_entity.text += (is_subword ? "" : " ") + clean_token;
                } else {
                    if (last_label_type != 0) entities.push_back(current_entity);
                    current_entity.text = clean_token;
                    current_entity.label = label_map[best_label];
                }
            } else {
                if (last_label_type != 0) entities.push_back(current_entity);
            }
            last_label_type = label_type;
        }
        if (last_label_type != 0) entities.push_back(current_entity);

        for (const auto & ent : entities) {
            ListVector::Reserve(result, current_offset + 1);
            FlatVector::GetData<string_t>(*entity_vector)[current_offset] = StringVector::AddString(*entity_vector, ent.text);
            FlatVector::GetData<string_t>(*label_vector)[current_offset] = StringVector::AddString(*label_vector, ent.label);
            current_offset++;
        }
        result_data[i].length = entities.size();
    }
    ListVector::SetListSize(result, current_offset);
    result.SetVectorType(VectorType::FLAT_VECTOR);
}

static void SetNerModelPath(ClientContext &context, SetScope scope, Value &parameter) {
    auto path = parameter.ToString();
    LoadModel(path);
}

static void LoadInternal(ExtensionLoader &loader) {
    auto & db = loader.GetDatabaseInstance();
    
    child_list_t<LogicalType> struct_children;
    struct_children.push_back(make_pair("entity", LogicalType::VARCHAR));
    struct_children.push_back(make_pair("label", LogicalType::VARCHAR));
    auto struct_type = LogicalType::STRUCT(struct_children);
    auto res_type = LogicalType::LIST(struct_type);

    // Register 'ner'
    ScalarFunctionSet ner_set("ner");
    ner_set.AddFunction(ScalarFunction({LogicalType::VARCHAR}, res_type, NerScalarFun));
    ner_set.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BOOLEAN}, res_type, NerScalarFun));
    for (auto & func : ner_set.functions) {
        func.stability = FunctionStability::VOLATILE;
    }
    loader.RegisterFunction(ner_set);

    // Register 'ner_extract' as alias
    ScalarFunctionSet ner_extract_set("ner_extract");
    ner_extract_set.AddFunction(ScalarFunction({LogicalType::VARCHAR}, res_type, NerScalarFun));
    ner_extract_set.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BOOLEAN}, res_type, NerScalarFun));
    for (auto & func : ner_extract_set.functions) {
        func.stability = FunctionStability::VOLATILE;
    }
    loader.RegisterFunction(ner_extract_set);

    auto &config = DBConfig::GetConfig(db);
    config.AddExtensionOption("ner_model_path", "Path to the NER model file", LogicalType::VARCHAR, Value(), SetNerModelPath);
}

void NerExtension::Load(ExtensionLoader &loader) {
    LoadInternal(loader);
}

std::string NerExtension::Name() { return "ner"; }

std::string NerExtension::Version() const {
#ifdef EXT_VERSION_NER
    return EXT_VERSION_NER;
#else
    return "";
#endif
}

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(ner, loader) {
    duckdb::LoadInternal(loader);
}
}
