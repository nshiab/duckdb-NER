# Named Entity Recognition (NER) DuckDB Extension

A high-performance Named Entity Recognition (NER) extension for DuckDB, powered by `ggml` and `bert.cpp`.

## Features

- **In-database Inference**: Extract entities directly from your DuckDB tables using SQL.
- **Lightweight**: Statically linked C++ with minimal dependencies.
- **BERT-based**: Supports state-of-the-art BERT NER models (CoNLL-2003 labels).
- **Flexible API**: Return results as `LIST(STRUCT(entity VARCHAR, label VARCHAR))`.

## Quick Start

### 1. Build the extension

```bash
make
```

### 2. Prepare the model

Since BERT models are large, they are not bundled with the binary. Use the provided script to download and convert a default model (`dslim/bert-base-NER`):

```bash
# Requires Python, torch, and transformers
./scripts/download_default_model.sh
```

### 3. Use in DuckDB

```sql
INSTALL ner;
LOAD ner;

-- Set path to the converted GGML model
SET ner_model_path = 'models/dslim_bert-base-NER_ner.bin';

-- Basic usage
SELECT ner('DuckDB Labs is located in Amsterdam');
-- [{'entity': 'DuckDB Labs', 'label': 'ORG'}, {'entity': 'Amsterdam', 'label': 'LOC'}]

-- Using the alias
SELECT ner_extract('Bill Gates co-founded Microsoft');
-- [{'entity': 'Bill Gates', 'label': 'PER'}, {'entity': 'Microsoft', 'label': 'ORG'}]

-- With truncation (default=true)
SELECT ner('Very long string...', true);
```

## API Reference

### `ner(text, [truncate])` / `ner_extract(text, [truncate])`

Extracts named entities from the given text.

- **Arguments**:
    - `text` (VARCHAR): The input string.
    - `truncate` (BOOLEAN, optional): If `true` (default), silently truncates input that exceeds the model's token limit (usually 512). If `false`, throws an error.
- **Returns**: `LIST(STRUCT(entity VARCHAR, label VARCHAR))`

### Settings

- `ner_model_path`: Path to the GGML model file (created via `scripts/convert_ner_to_ggml.py`).

## Label Mapping

The default model uses the following labels:
- `PER`: Person
- `ORG`: Organization
- `LOC`: Location
- `MISC`: Miscellaneous

## Acknowledgements

Built on top of [ggml](https://github.com/ggerganov/ggml) and [bert.cpp](https://github.com/skeskinen/bert.cpp).
