#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install torch transformers numpy

echo "Downloading and converting dslim/bert-base-NER to GGML format..."
python3 scripts/convert_ner_to_ggml.py dslim/bert-base-NER

echo ""
echo "Model prepared! You can now use it in DuckDB with:"
echo "SET ner_model_path = 'models/dslim_bert-base-NER_ner.bin';"
