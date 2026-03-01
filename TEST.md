# Testing & Verification

## Functional Testing

Functional tests are implemented using DuckDB's `SQLLogicTest` framework. The tests cover:
- Extension loading and configuration.
- Function registration (`ner` and `ner_extract`).
- Handling of missing models.
- Positional argument handling for the `truncate` parameter.
- Basic API structure verification (`LIST(STRUCT)`).

To run the tests:
```bash
make test
```

## Model Integration Verification

Verification of the inference engine was performed by:
1. Validating the GGUF loader with custom BERT-NER headers.
2. Implementing full Transformer blocks in `src/ner_model.cpp` and cross-referencing with `bert.cpp`.
3. Implementing entity reconstruction (BIO tagging merging) and WordPiece token reconstruction.

## Performance Considerations

- **Inference**: Performance is highly dependent on the number of tokens and the model size. `bert-base` models (~110M parameters) provide a good balance between accuracy and speed.
- **Memory**: The extension uses a pre-allocated compute buffer (128MB by default) to minimize runtime allocations.
- **Parallelism**: The `ner_eval` function supports multi-threaded execution (configured to 4 threads by default in the current implementation).

## Accuracy

Accuracy depends on the model loaded via `ner_model_path`. The provided conversion script supports standard Hugging Face models like `dslim/bert-base-NER`, which is fine-tuned on the CoNLL-2003 dataset.
