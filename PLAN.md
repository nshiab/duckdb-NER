# Named Entity Recognition (NER) DuckDB Extension

## Status

- [x] **Milestone 1: Scaffold & CI Setup**
    - [x] Initialized from DuckDB extension template.
    - [x] Renamed to `ner`.
    - [x] Integrated `bert.cpp` and `ggml` as submodules.
    - [x] Updated CI workflows (`ExtensionTemplate.yml`, `MainDistributionPipeline.yml`) to support recursive submodules and renaming.
    - [x] Verified build and basic test with `LIST(STRUCT)` return type.

- [x] **Milestone 2: Model Selection & Integration**
    - [x] Implemented `ner_model.cpp` and `ner_model.hpp` (adapted from `bert.cpp`).
    - [x] Added support for token classification (NER) head in the loader and inference.
    - [x] Created `scripts/convert_ner_to_ggml.py` for BERT-NER models.
    - [x] Created `scripts/download_default_model.sh` for easy setup.
    - [x] Complete the full Transformer block implementation in `ner_eval`.
    - [x] Verified build with real Transformer logic.

- [x] **Milestone 3: API & UX Refinement**
    - [x] Implement `truncate=true` logic.
    - [x] Refine entity reconstruction (WordPiece merging and BIO tagging).
    - [x] Support custom model loading via `SET ner_model_path`.
    - [x] Added `ner_extract` alias.

- [x] **Milestone 4: Robust Testing**
    - [x] Test with SQL logic tests.
    - [x] Verify settings and alias.

- [x] **Milestone 5: Documentation**
    - [x] Finalize `README.md` and `TEST.md`.

## Architectural Decisions

1. **Inference Engine**: Using `ggml` via a custom adaptation of `bert.cpp` (`src/ner_model.cpp`). This ensures static linking, high performance on CPU, and minimal dependencies.
2. **Model Format**: Custom GGML-based format that includes BERT-NER classification head weights.
3. **Model Distribution**: Models are not bundled in the binary to keep it lightweight. A download script is provided, and users can load models from any path using a DuckDB setting.
4. **Return Type**: `LIST(STRUCT(entity VARCHAR, label VARCHAR))` for seamless SQL integration.
