import pickle
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from model.transformer_model import Transformer
from huggingface_hub import hf_hub_download

# --- NEW: seeding for reproducibility ---------------------------------------
import os, random
try:
    import numpy as np
except ImportError:
    np = None

SEED = 42  # <- set your fixed seed here

# Seed Python and NumPy (if available)
random.seed(SEED)
if np is not None:
    np.random.seed(SEED)

# Seed PyTorch (CPU and, if present, GPU)
torch.manual_seed(SEED)
# cuDNN determinism (only relevant when CUDA backend is used)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ----------------------------------------------------------------------------

repo_id = "amaai-lab/text2midi"
# Download the model.bin file
model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
# Download the vocab_remi.pkl file
tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")

# Load the tokenizer dictionary
with open(tokenizer_path, "rb") as f:
    r_tokenizer = pickle.load(f)

# Get the vocab size
vocab_size = len(r_tokenizer)
print("Vocab size: ", vocab_size)
model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

print('Model loaded.')

# Enter the text prompt and tokenize it
src = "A melodic electronic song with ambient elements, featuring piano, acoustic guitar, alto saxophone, string ensemble, and electric bass. Set in G minor with a 4/4 time signature, it moves at a lively Presto tempo. The composition evokes a blend of relaxation and darkness, with hints of happiness and a meditative quality."
print('Generating for prompt: ' + src)

inputs = tokenizer(src, return_tensors='pt', padding=True, truncation=True)
input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
input_ids = input_ids.to(device)
attention_mask = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0)
attention_mask = attention_mask.to(device)

# Generate the midi (unchanged)
output = model.generate(input_ids, attention_mask, max_len=2000, temperature=1.0)
output_list = output[0].tolist()
generated_midi = r_tokenizer.decode(output_list)
generated_midi.dump_midi("output.mid")
