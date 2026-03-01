# Named Entity Recognition (NER) DuckDB Extension

1. Architectural Decisions Building a statically linked, cross-platform ML
   extension in DuckDB requires strict dependency management. We cannot rely on
   Python or heavy dynamic libraries.

A. C++ Inference Engine The Challenge: Frameworks like PyTorch or standard ONNX
Runtime are too heavy and notoriously difficult to cross-compile for WebAssembly
(WASM) and ARM architectures within DuckDB's GitHub Actions matrix.

The Solution: We will use ggml (the tensor library behind llama.cpp). It is
written in pure C, requires zero external dependencies, and compiles seamlessly
across CPU architectures, including WASM.

The Model: We will target a heavily quantized, lightweight BERT or DistilRoBERTa
model fine-tuned for token classification (NER).

B. Tokenization The Challenge: Transformer models require WordPiece or BPE
tokenization.

The Solution: We will implement a lightweight, pure C++ WordPiece tokenizer
within the extension.

C. Model Weight Distribution The Challenge: Bundling a 30MB+ model directly into
the .duckdb_extension binary will bloat the extension and cause memory issues,
especially in WASM.

The Solution: 1. The extension will ship without bundled weights. 2. On first
invocation of ner(), if no custom model is set, it will automatically download
the default lightweight quantized model (hosted on HuggingFace or a static
GitHub release) to the local ~/.duckdb/extensions/ner_models/ directory. 3. The
model will be loaded into memory once per connection/session.

2. DuckDB API Design Data Types Input: VARCHAR (The text to analyze).

Output: LIST(STRUCT(entity VARCHAR, label VARCHAR)).

Functions & PRAGMAs The Core Scalar Function: * ner(VARCHAR, truncate BOOLEAN
DEFAULT true)

Behavior: If truncate=true, strings exceeding the model's context window (e.g.,
512 tokens) will be silently truncated. If false, it throws an out-of-bounds
error or processes in chunks (chunking is preferred but more complex).

Configuration Settings:

SET ner_model_path = '/path/to/custom/model.bin';

SET ner_confidence_threshold = 0.7;

3. CI/CD & Build Strategy DuckDB's GitHub Actions matrix is unforgiving. If the
   build fails on Windows or WASM, the extension fails.

Dependencies: ggml will be added as a Git submodule. We will avoid vcpkg if
possible to keep the CI fast and reliable.

CMakeLists.txt: We will modify the DuckDB extension template's CMake file to
build ggml statically and link it directly into our extension binary.

WASM Fallback: If ggml proves too resource-intensive for the WASM build matrix
during Phase 2, we will conditionally disable the WASM target or fall back to a
tiny statistical CRF (Conditional Random Field) model exclusively for the web
build.

4. Development Milestones

[ ] Milestone 1: Scaffold & CI Setup

Initialize the template, add the ggml submodule.

Modify CMakeLists.txt and ensure the empty extension compiles across all GitHub
Actions (Linux, macOS, Windows).

[ ] Milestone 2: Tokenizer & API Boilerplate

Implement the DuckDB ner() scalar function returning the LIST(STRUCT(...)) mock
data.

Implement the C++ WordPiece tokenizer.

Add the ner_model_path setting variable.

Verification: CI passes.

[ ] Milestone 3: Model Inference Integration

Integrate ggml to load a quantized DistilBERT binary.

Map token probabilities to NER tags (B-ORG, I-ORG, B-PER, etc.).

Verification: CI passes.

[ ] Milestone 4: Auto-Download & Testing

Implement the HTTP client logic (using DuckDB's internal HTTP tools if possible,
or lightweight cpp-httplib) to fetch the default model on first run.

Write SQL tests for accuracy (known strings) and truncation behavior.

Verification: CI passes.

[ ] Milestone 5: Benchmarking & Documentation

Run performance tests on a 1M+ row table.

Generate TEST.md and finalize README.md.
