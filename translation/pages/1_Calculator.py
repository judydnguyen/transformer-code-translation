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
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)

import inspect

from code_editor import code_editor


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

print("Finished loading the model!")

def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples,total=len(examples))):
        ##extract data flow
        code_tokens,dfg=extract_dataflow(example.source,
                                         parsers["c_sharp" if args.source_lang == "cs" else "java"],
                                         "c_sharp" if args.source_lang == "cs" else "java")
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        
        #truncating
        code_tokens=code_tokens[:args.max_source_length-3][:512-3]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:args.max_source_length-len(source_tokens)]
        source_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.max_source_length-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length      
        source_mask = [1] * (len(source_tokens))
        source_mask+=[0]*padding_length        
        
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
      

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                logger.info("position_idx: {}".format(position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, dfg_to_dfg))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features


def model_infer(sample_code):
    """Use the code translation model

    :param sample_code: source code chunk
    :type sample_code: string
    :return: target code chunk
    :rtype: string
    """
    # Tokenize the input
    source_ids = tokenizer.encode(java_code, return_tensors="pt", truncation=True, max_length=512).to('cpu')
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

    return translated_code

####################
## STREAMLIT DEMO ##
####################

# This is the code chunk we will use...
#
# java_body = """
#   public class Calculator {
#       public static int add(int a, int b) {
#           return a + b;    
#       }
# } 
# """

# Inject custom CSS
st.markdown(
    """
    <style>
    .custom-text {
        font-size: 20px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use the custom CSS class
st.markdown('<div class="custom-text">We will start with a simple example: computing the sum of two integers.</div>', unsafe_allow_html=True)
st.markdown('')

st.header("Source Java Code")

# Add a button with text: 'Save'
custom_btns = [{"name": "Save", "hasText": True, "alwaysOn": True, "commands": ["submit"], "style": {"top": "0.46rem", "right": "0.4rem"}}]

# Set up a code editor for the user to add code
code_input = code_editor(
    "// Write your Java code here", 
    lang="java",
    height=10,
    buttons=custom_btns
)

# Retrieve the Java code from the code editor
java_code = code_input["text"]

st.header("Translated C# Code")

# Button to process the input code
if st.button("Translate"):
    # Call the function and save the output
    output = model_infer(java_code)
    
    # Display the prediction
    code_output = code_editor(
        output, 
        lang="csharp",
        height=10,
    )   

    # Add the expected code
    st.header("Target C# Code")
    expected_code_output = code_editor(
        "public static int Add(int a, int b) { return a + b; }", lang="csharp"
    )

#
# If you want to verify, plug this snippet into:
# https://codapi.org/csharp/ to run it.
# 
# using System;
#
# // static long Factorial(int n) => n <= 1 ? 1 : n * Factorial(n - 1);
# static long Factorial(int n){if (n <= 1){return 1;}return n * Factorial(n - 1);}
#
# long output = Factorial(3);
#
# Console.WriteLine(output);
#
