import sys
import struct
import json
import torch
import numpy as np
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer

if len(sys.argv) < 2:
    print("Usage: convert_ner_to_ggml.py model_name_or_path [use-f32]
")
    sys.exit(1)

model_id = sys.argv[1]
ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])

print(f"Loading model {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id, low_cpu_mem_usage=True)

# Ensure it's a BertForTokenClassification model
if model.config.model_type != "bert":
    print(f"Error: Only BERT models are supported, got {model.config.model_type}")
    sys.exit(1)

hparams = model.config.to_dict()
list_vars = model.state_dict()

# Create model directory if it doesn't exist
os.makedirs("models", exist_ok=True)
fname_out = f"models/{model_id.replace('/', '_')}_ner.bin"

fout = open(fname_out, "wb")

# Header
fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["max_position_embeddings"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["intermediate_size"]))
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
fout.write(struct.pack("i", ftype))
fout.write(struct.pack("i", hparams["num_labels"])) # Added for NER

# Vocab
vocab = tokenizer.get_vocab()
# Sort by ID
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
for word, _ in sorted_vocab:
    data = bytes(word, 'utf-8')
    fout.write(struct.pack("i", len(data)))
    fout.write(data)

# Tensors
for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    
    # Map HF tensor names to our internal names if needed, 
    # but here we use names like 'embeddings.word_embeddings.weight'
    # which match our loader.
    
    # We need to handle BERT-specific naming
    clean_name = name
    if name.startswith("bert."):
        clean_name = name[5:]
    
    if clean_name in ['embeddings.position_ids']:
        continue

    print(f"Writing tensor {clean_name} with shape {data.shape}")

    n_dims = len(data.shape)
    l_type = 0
    if ftype == 1 and clean_name.endswith(".weight") and n_dims == 2:
        data = data.astype(np.float16)
        l_type = 1
    else:
        data = data.astype(np.float32)
        l_type = 0

    str_name = clean_name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str_name), l_type))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str_name)
    data.tofile(fout)

fout.close()
print(f"Done! Model saved to {fname_out}")
