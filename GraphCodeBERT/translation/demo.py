from __future__ import absolute_import
import streamlit as st

####################
###### HELPER ######
####################
import requests
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

import inspect


# GLOBAL VARIABLES
source="java"
target="cs"
lr=1e-4
batch_size=64
beam_size=10
source_length=320
target_length=256
output_dir=f"saved_models/{source}-{target}/"
train_file=f"data/train.java-cs.txt.{source},data/train.java-cs.txt.{target}"
dev_file=f"data/valid.java-cs.txt.{source},data/valid.java-cs.txt.{target}"
epochs=2
pretrained_model="microsoft/graphcodebert-base"

# URL of the file
url = "https://huggingface.co/judynguyen16/graphcodebert-code-translation-java-cs/resolve/main/pytorch_model.bin"

# Path to save the file
save_path = "pytorch_model.bin"

# Download the file (if it does not exist)
if not os.path.exists(save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded and saved to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# Simulate argparse for Jupyter
class Args:
    model_type = "roberta"
    model_name_or_path = "roberta-base"
    output_dir = "./output"
    load_model_path = None
    train_filename = None
    dev_filename = None
    test_filename = None
    source_lang = "en"
    config_name = ""
    tokenizer_name = ""
    max_source_length = 64
    max_target_length = 32
    do_train = True
    do_eval = True
    do_test = False
    do_lower_case = False
    # no_cuda = False
    no_cuda = True
    train_batch_size = 256
    eval_batch_size = 512
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    beam_size = 10
    weight_decay = 0.0
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    num_train_epochs = 3
    max_steps = -1
    eval_steps = -1
    train_steps = -1
    warmup_steps = 0
    local_rank = -1
    seed = 42

# Create an instance of Args
args = Args()

# Print the arguments for verification
print(f"Arguments: {args}")
args.model_type = "roberta"
args.source_lang = source
args.target_lang = target
args.model_name_or_path = pretrained_model
args.tokenizer_name = "microsoft/graphcodebert-base"
args.config_name = "microsoft/graphcodebert-base"
args.train_filename = train_file
args.dev_filename = dev_file
args.output_dir = output_dir
args.learning_rate = lr
args.num_train_epochs = epochs
args.train_batch_size = batch_size
args.eval_batch_size = batch_size
args.max_source_length = source_length
args.max_target_length = target_length

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name)
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name )

# build model
encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                beam_size=args.beam_size,max_length=args.max_target_length,
                sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

print("Trying to load the model.")

# load the model and set to eval
model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
model.to('cpu')
model.eval()

# print(inspect.signature(model.forward))

java_body = """
public static long factorial(int n) { 
    if (n <= 1) { return 1; } return n * factorial(n - 1); 
}
"""

# Tokenize the input
source_ids = tokenizer.encode(java_body, return_tensors="pt", truncation=True, max_length=512).to('cpu')
source_mask = source_ids.ne(tokenizer.pad_token_id).to('cpu')  # Boolean mask
position_idx = torch.arange(0, source_ids.size(1), dtype=torch.long).unsqueeze(0).to('cpu')  # Positional indices
attn_mask = source_mask.clone()  # Copy the same mask if no special processing is required

# Perform inference
with torch.no_grad():
    outputs = model(
        source_ids=source_ids,
        source_mask=source_mask,
        position_idx=position_idx,
        attn_mask=attn_mask
    )

# Decode the output
generated_ids = outputs[0]  # Assuming the first output contains generated token IDs
translated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Finished loading the model!")

####################
## STREAMLIT DEMO ##
####################

st.title("Code Translation Demo")

st.text("We demonstrate the ability of the model to translate a chunk of Java code to C# using an abstract syntax tree to augment the translation.")

st.header("Java Code")

st.code(java_body, language="java")

st.header("C# Code")

cs_body = """
    public static long Factorial(int n) => n <= 1 ? 1 : n * Factorial(n - 1);
"""
st.code(cs_body, language="csharp")

st.header("Predicted C# Code")
st.code(translated_code, language="csharp")

