# Named Entity Recognition (NER) DuckDB Extension

A high-performance Named Entity Recognition (NER) extension for DuckDB, powered by `ggml` and `bert.cpp`.

## Features

- **Out-of-the-box Inference**: Works immediately with a bundled tiny model.
- **In-database Inference**: Extract entities directly from your DuckDB tables using SQL.
- **Lightweight**: Statically linked C++ with minimal dependencies.
- **Wasm Compatible**: Bundled model makes it easier to use in DuckDB-Wasm.
- **Flexible API**: Return results as `LIST(STRUCT(entity VARCHAR, label VARCHAR))`.

## Quick Start

### 1. Build the extension

```bash
make
```

### 2. Use in DuckDB

```sql
LOAD ner;

-- Basic usage (uses bundled tiny model by default)
SELECT ner('DuckDB Labs is located in Amsterdam');
```

### 3. (Optional) Use a larger/custom model

To use a more accurate model (like `bert-base-NER`), download and convert it:

```bash
./scripts/download_default_model.sh
```

Then load it in DuckDB:

```sql
SET ner_model_path = 'models/dslim_bert-base-NER_ner.bin';
SELECT ner('DuckDB Labs is located in Amsterdam');
```

## API Reference

### `ner(text, [truncate])` / `ner_extract(text, [truncate])`

Extracts named entities from the given text.

- **Arguments**:
    - `text` (VARCHAR): The input string.
    - `truncate` (BOOLEAN, optional): If `true` (default), silently truncates input that exceeds the model's token limit (usually 512). If `false`, throws an error.
- **Returns**: `LIST(STRUCT(entity VARCHAR, label VARCHAR))`

### Settings

- `ner_model_path`: Path to a GGML model file (overrides the bundled model).

## Acknowledgements

Built on top of [ggml](https://github.com/ggerganov/ggml) and [bert.cpp](https://github.com/skeskinen/bert.cpp).
