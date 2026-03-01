#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install torch transformers numpy

# 1. Download and convert large model for user reference
echo "Downloading and converting dslim/bert-base-NER to GGML format..."
python3 scripts/convert_ner_to_ggml.py dslim/bert-base-NER

# 2. Download and convert a tiny model to bundle as default
# We use a tiny BERT model fine-tuned for NER if available, or just a tiny BERT.
# For this example, let's assume we want to bundle dslim/bert-tiny-ner if it exists,
# or we convert a tiny one.
echo "Downloading and converting a tiny model for bundling..."
# Using a small one for demonstration. In a real production, you'd use a specific fine-tuned tiny NER.
python3 scripts/convert_ner_to_ggml.py prajjwal1/bert-tiny

echo "Generating C++ header for bundled model..."
python3 scripts/generate_model_header.py models/prajjwal1_bert-tiny_ner.bin src/include/default_model.hpp

echo ""
echo "Model preparation complete!"
echo "Large model: models/dslim_bert-base-NER_ner.bin"
echo "Tiny model (bundled): src/include/default_model.hpp"
echo ""
echo "You can now rebuild the extension to include the tiny model by default."
echo "Usage in DuckDB: SELECT ner('text'); -- uses bundled tiny model"
