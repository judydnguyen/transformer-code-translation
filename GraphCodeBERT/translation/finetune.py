# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
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
from termcolor import colored
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
np.bool = np.bool_
# https://github.com/grantjenks/py-tree-sitter-languages/issues/64
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'c_sharp':DFG_csharp,
}

logger = logging.getLogger(__name__)
#load parsers
parsers={}        
for lang in dfg_function:
    # print(Language)
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

def extract_controlflow(code, parser, lang):
    """
    Extract control flow edges (CFG) from the code.
    """
    try:
        # Parse the code to obtain the AST
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node

        # Extract control flow edges from the AST
        cfg = extract_control_flow_edges(root_node, tokens_index=tree_to_token_index(root_node))
        return cfg
    except Exception as e:
        print(f"Error extracting CFG: {e}")
        return []


# def extract_control_flow_edges(root_node):
#     """
#     Extract control flow edges (e.g., branching, loops) from the AST.
#     """
#     edges = []

#     def traverse(node):
#         if node.type == "if_statement":
#             # Extract the condition
#             condition_node = next(
#                 (child for child in node.children if child.type == "parenthesized_expression"), None
#             )
#             # Extract the "then" block
#             then_node = next(
#                 (child for child in node.children if child.type == "block"), None
#             )
#             # Add edges for condition -> then
#             if condition_node and then_node:
#                 edges.append((condition_node.start_point, then_node.start_point))
#             # Extract the "else" block, if present
#             else_node = next(
#                 (child for child in node.children if child.type == "else_body"), None
#             )
#             if condition_node and else_node:
#                 edges.append((condition_node.start_point, else_node.start_point))
#         elif node.type in ("for_statement", "while_statement"):
#             # Extract loop condition and body
#             condition_node = next(
#                 (child for child in node.children if child.type == "parenthesized_expression"), None
#             )
#             body_node = next(
#                 (child for child in node.children if child.type == "block"), None
#             )
#             if condition_node and body_node:
#                 edges.append((condition_node.start_point, body_node.start_point))
#         elif node.type == "return_statement":
#             # Add a return statement as a flow edge
#             edges.append((node.start_point, node.end_point))

#         # Recurse for all children
#         for child in node.children:
#             traverse(child)

#     traverse(root_node)
#     return edges
def extract_control_flow_edges(root_node, tokens_index):
    """
    Extract control flow edges (e.g., branching, loops) from the AST,
    mapping node positions to token indices.
    
    Args:
        root_node: The root node of the AST.
        tokens_index: A list of token positions derived from tree_to_token_index.
    
    Returns:
        edges: A list of control flow edges with token indices and node types.
    """
    edges = []

    def map_position_to_token_index(position):
        # Map a position (start_point or end_point) to a token index
        for idx, (start, end) in enumerate(tokens_index):
            if start <= position < end:
                return idx
        return -1  # Return -1 if position is not found

    def traverse(node):
        if node.type == "if_statement":
            # Extract the condition
            condition_node = next(
                (child for child in node.children if child.type == "parenthesized_expression"), None
            )
            # Extract the "then" block
            then_node = next(
                (child for child in node.children if child.type == "block"), None
            )
            # Add edges for condition -> then
            if condition_node and then_node:
                edges.append((
                    (map_position_to_token_index(condition_node.start_point), condition_node.type),
                    (map_position_to_token_index(then_node.start_point), then_node.type)
                ))
            # Extract the "else" block, if present
            else_node = next(
                (child for child in node.children if child.type == "else_body"), None
            )
            if condition_node and else_node:
                edges.append((
                    (map_position_to_token_index(condition_node.start_point), condition_node.type),
                    (map_position_to_token_index(else_node.start_point), else_node.type)
                ))
        elif node.type in ("for_statement", "while_statement"):
            # Extract loop condition and body
            condition_node = next(
                (child for child in node.children if child.type == "parenthesized_expression"), None
            )
            body_node = next(
                (child for child in node.children if child.type == "block"), None
            )
            if condition_node and body_node:
                edges.append((
                    (map_position_to_token_index(condition_node.start_point), condition_node.type),
                    (map_position_to_token_index(body_node.start_point), body_node.type)
                ))
        elif node.type == "return_statement":
            # Add a return statement as a flow edge
            edges.append((
                (map_position_to_token_index(node.start_point), node.type),
                (map_position_to_token_index(node.end_point), node.type)
            ))

        # Recurse for all children
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return edges

    
#remove comments, tokenize code and extract dataflow     
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


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 source,
                 target,
                 lang
                 ):
        self.source = source
        self.target = target
        self.lang=lang

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    source,target=filename.split(',')
    lang='java'
    if source[-1]=='s':
        lang='c_sharp'
        
    with open(source,encoding="utf-8") as f1,open(target,encoding="utf-8") as f2:
        for line1,line2 in zip(f1,f2):
            line1=line1.strip()
            line2=line2.strip()
            examples.append(
                Example(
                    source=line1,
                    target=line2,
                    lang=lang
                        ) 
            )

    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 cfg_to_code,
                 dfg_to_dfg,                 
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.cfg_to_code = cfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
# def convert_examples_to_features(examples, tokenizer, args,stage=None):
#     features = []
#     for example_index, example in enumerate(tqdm(examples,total=len(examples))):
#         ##extract data flow
#         code_tokens,dfg=extract_dataflow(example.source,
#                                          parsers["c_sharp" if args.source_lang == "cs" else "java"],
#                                          "c_sharp" if args.source_lang == "cs" else "java")
#         cfg = extract_controlflow(example.source,
#                                     parsers["c_sharp" if args.source_lang == "cs" else "java"],
#                                     "c_sharp" if args.source_lang == "cs" else "java")
        
#         code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
#         ori2cur_pos={}
#         ori2cur_pos[-1]=(0,0)
#         for i in range(len(code_tokens)):
#             ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
#         code_tokens=[y for x in code_tokens for y in x]  
        
#         #truncating
#         code_tokens=code_tokens[:args.max_source_length-3][:512-3]
#         source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
#         source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
#         position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        
#         dfg=dfg[:args.max_source_length-len(source_tokens)]
        
#         source_tokens+=[x[0] for x in dfg]
#         position_idx+=[0 for x in dfg]
#         source_ids+=[tokenizer.unk_token_id for x in dfg]
        
#         # ensure the length of source_ids is less than args.max_source_length
#         if len(source_ids) < args.max_source_length:
#             # clip cfg to fit the source length
#             cfg = cfg[:args.max_source_length-len(source_ids)]
            
#             # Integrate CFG into input features
#             for edge in cfg:
#                 source, dest = edge
#                 source_tokens.append(source[1])  # Append source node type (e.g., 'parenthesized_expression')
#                 source_tokens.append(dest[1])    # Append destination node type (e.g., 'block')
#                 position_idx.append(0)           # Assign positions (or compute them based on AST structure if needed)
#                 position_idx.append(0)
#                 source_ids.append(tokenizer.unk_token_id)
#                 # source_ids.append(tokenizer.unk_token_id)
#             # try to print the source_tokens for the cfg
#             # print(f"source_tokens: {source_tokens}")
        
#         # return
#         padding_length=args.max_source_length-len(source_ids)
#         position_idx+=[tokenizer.pad_token_id]*padding_length
#         source_ids+=[tokenizer.pad_token_id]*padding_length      
#         source_mask = [1] * (len(source_tokens))
#         source_mask +=[0]*padding_length        
        
#         # reindex
#         reverse_index={}
#         for idx,x in enumerate(dfg):
#             reverse_index[x[1]]=idx
#         for idx,x in enumerate(dfg):
#             dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
#         dfg_to_dfg=[x[-1] for x in dfg]
        
#         dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
#         length=len([tokenizer.cls_token])
#         dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code] 

#         # Include CFG in reindexing
#         cfg_to_code = []
        
#         for edge in cfg:
#             source, dest = edge
#             # Debugging
#             # import IPython; IPython.embed()
            
#             # Assuming ori2cur_pos provides mappings to tokenized positions
#             # check if nodes going over the length of the source_ids

#             src_pos = ori2cur_pos[source[0]]
#             dest_pos = ori2cur_pos[dest[0]]
#             if src_pos[1] >= args.max_source_length - 3 or dest_pos[1] >= args.max_source_length -3:
#                 print(f"skipped, Source: {source}, Dest: {dest}")
#                 continue
#             # Append the tokenized positions with offset (length)
#             # cfg_to_code.append((src_pos[0] + length, dest_pos[1] + length))
            
#             cfg_to_code.append((src_pos[0]+length, src_pos[1]+length))
#             cfg_to_code.append((dest_pos[0]+length, dest_pos[1]+length))


#         #target
#         if stage=="test":
#             target_tokens = tokenizer.tokenize("None")
#         else:
#             target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
#         target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
#         target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
#         target_mask = [1] *len(target_ids)
#         padding_length = args.max_target_length - len(target_ids)
#         target_ids+=[tokenizer.pad_token_id]*padding_length
#         target_mask+=[0]*padding_length   
   
#         if example_index < 5:
#             if stage=='train':
#                 logger.info("*** Example ***")
#                 logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
#                 logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
#                 logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
#                 logger.info("position_idx: {}".format(position_idx))
#                 logger.info("dfg_to_code: {}".format(' '.join(map(str, dfg_to_code))))
#                 logger.info("dfg_to_dfg: {}".format(' '.join(map(str, dfg_to_dfg))))
                
#                 logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
#                 logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
#                 logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
#         features.append(
#             InputFeatures(
#                  example_index,
#                  source_ids,
#                  position_idx,
#                  dfg_to_code,
#                  cfg_to_code,
#                  dfg_to_dfg,
#                  target_ids,
#                  source_mask,
#                  target_mask,
#             )
#         )
#     return features

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples, total=len(examples))):
        # Extract data flow (DFG) and control flow (CFG)
        code_tokens, dfg = extract_dataflow(
            example.source,
            parsers["c_sharp" if args.source_lang == "cs" else "java"],
            "c_sharp" if args.source_lang == "cs" else "java",
        )
        cfg = extract_controlflow(
            example.source,
            parsers["c_sharp" if args.source_lang == "cs" else "java"],
            "c_sharp" if args.source_lang == "cs" else "java",
        )

        # Tokenize the code
        code_tokens = [
            tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x)
            for idx, x in enumerate(code_tokens)
        ]
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]

        # Truncate code tokens
        code_tokens = code_tokens[:args.max_source_length - 3][:512 - 3]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

        # Integrate DFG into input features
        dfg = dfg[:args.max_source_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        source_ids += [tokenizer.unk_token_id for x in dfg]

        # Truncate CFG if needed
        if len(source_ids) < args.max_source_length:
            filtered_cfg = []
            for edge in cfg:
                source, dest = edge
                src_pos = ori2cur_pos[source[0]]
                dest_pos = ori2cur_pos[dest[0]]
                if src_pos[1] < args.max_source_length - 3 and dest_pos[1] < args.max_source_length - 3:
                    filtered_cfg.append(edge)
                else:
                    print(f"Skipped edge due to position overflow: Source: {source}, Dest: {dest}")
            cfg = filtered_cfg

            # Add CFG nodes to source tokens
            for edge in cfg:
                source, dest = edge
                src_pos = ori2cur_pos[source[0]]
                dest_pos = ori2cur_pos[dest[0]]
                if len(source_tokens) + 2 >= args.max_source_length:
                    print(f"Skipped edge due to token overflow: Source: {source}, Dest: {dest}")
                    break
                source_tokens.append(source[1])
                source_tokens.append(dest[1])
                position_idx.append(0)
                position_idx.append(0)
                source_ids.append(tokenizer.unk_token_id)
                source_ids.append(tokenizer.unk_token_id)

        # Pad to max_source_length
        padding_length = args.max_source_length - len(source_ids)
        if padding_length < 0:
            print(f"Source ID Length: {len(source_ids)}")
            print(f"Source Tokens Length: {len(source_tokens)}")
            print(f"DFG Length: {len(dfg)}")
            print(f"CFG Length: {len(cfg)}")
            print(f"Position Index Length: {len(position_idx)}")
            print(f"Source Tokens: {source_tokens}")
            print(f"DFG: {dfg}")
            print(f"CFG: {cfg}")
            print(f"Position Index: {position_idx}")
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask = [1] * len(source_tokens) + [0] * padding_length

        # Reindex DFG and CFG
        reverse_index = {x[1]: idx for idx, x in enumerate(dfg)}
        dfg = [x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],) for x in dfg]
        
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [(ori2cur_pos[x[1]][0] + len([tokenizer.cls_token]),
                        ori2cur_pos[x[1]][1] + len([tokenizer.cls_token]))
                       for x in dfg]

        cfg_to_code = []
        for edge in cfg:
            source, dest = edge
            src_pos = ori2cur_pos[source[0]]
            dest_pos = ori2cur_pos[dest[0]]
            cfg_to_code.append((src_pos[0] + len([tokenizer.cls_token]), src_pos[1] + len([tokenizer.cls_token])))
            cfg_to_code.append((dest_pos[0] + len([tokenizer.cls_token]), dest_pos[1] + len([tokenizer.cls_token])))

        # Handle target tokens
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
            
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        
        # if len(source_ids) != len(source_mask) or len(source_ids) != len(position_idx) or len(source_mask) > args.max_source_length:
        #     print(f"Source ID Length: {len(source_ids)}")
        #     print(f"Source Mask Length: {len(source_mask)}")
        assert len(source_ids) == len(source_mask) == len(position_idx) <= args.max_source_length

        # Log example details for debugging
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
#                 logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
        # check if the source_ids is less than the max_source_length
        if len(source_ids) < args.max_source_length:
            print(colored(f"Source ID Length: {len(source_ids)}", "red"))
            print(f"Source ID Length: {len(source_ids)}")
            print(f"Source Tokens Length: {len(source_tokens)}")
        # Append features
        features.append(
            InputFeatures(
                example_index,
                source_ids,
                position_idx,
                dfg_to_code,
                cfg_to_code,
                dfg_to_dfg,
                target_ids,
                source_mask,
                target_mask,
            )
        )

    return features

class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args=args  
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # Initialize attention mask
        attn_mask = np.zeros((self.args.max_source_length, self.args.max_source_length), dtype=np.bool)

        # Calculate node index and max length
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])

        # Sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True

        # Special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].source_ids):
            if i in [0, 2]:  # CLS, SEP tokens
                attn_mask[idx, :max_length] = True

        # Nodes attend to code tokens (DFG)
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True


        # Nodes attend to adjacent nodes (DFG)
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < self.args.max_source_length:
                    attn_mask[idx + node_index, a + node_index] = True

        # Nodes attend to code tokens (CFG)
        for idx, (a, b) in enumerate(self.examples[item].cfg_to_code):
            if a < node_index and b < node_index and idx + node_index < self.args.max_source_length:
                a = min(a, self.args.max_source_length - 1)
                b = min(b, self.args.max_source_length)
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True

        # Pad features to max_source_length
        padding_length = self.args.max_source_length - len(self.examples[item].source_ids)
        source_ids = self.examples[item].source_ids + [0] * padding_length
        
        source_mask = self.examples[item].source_mask + [0] * padding_length
        position_idx = self.examples[item].position_idx + [0] * padding_length

        # Return tensors
        return (
            torch.tensor(source_ids),
            torch.tensor(source_mask),
            torch.tensor(position_idx),
            torch.tensor(attn_mask),
            torch.tensor(self.examples[item].target_ids),
            torch.tensor(self.examples[item].target_mask),
        )

    
    # def __getitem__(self, item):
    #     #calculate graph-guided masked function
    #     attn_mask=np.zeros((self.args.max_source_length,self.args.max_source_length),dtype=np.bool)
    #     #calculate begin index of node and max length of input
    #     node_index=sum([i>1 for i in self.examples[item].position_idx])
    #     max_length=sum([i!=1 for i in self.examples[item].position_idx])
    #     # import IPython; IPython.embed()
    #     print(f"Node Index: {node_index}, Max Length: {max_length}")
    #     # return
    #     #sequence can attend to sequence
    #     attn_mask[:node_index,:node_index]=True
    #     #special tokens attend to all tokens
    #     for idx,i in enumerate(self.examples[item].source_ids):
    #         if i in [0,2]:
    #             attn_mask[idx,:max_length]=True
                
    #     #nodes attend to code tokens that are identified from
    #     for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
    #         if a<node_index and b<node_index:
    #             attn_mask[idx+node_index,a:b]=True
    #             attn_mask[a:b,idx+node_index]=True

                
    #     # #nodes attend to control flow nodes
    #     # for idx,(a,b) in enumerate(self.examples[item].cfg_to_code):
    #     #     if a<node_index and b<node_index:
    #     #         attn_mask[idx+node_index,a:b]=True
    #     #         attn_mask[a:b,idx+node_index]=True
                
    #     #nodes attend to adjacent nodes         
    #     for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
    #         for a in nodes:
    #             if a+node_index<len(self.examples[item].position_idx):
    #                 attn_mask[idx+node_index,a+node_index]=True  

    #     for idx, (a, b) in enumerate(self.examples[item].cfg_to_code):
    #         if a < node_index and b < node_index and idx + node_index < max_length:
    #             try :
    #                 attn_mask[idx + node_index, a:b] = True
    #                 attn_mask[a:b, idx + node_index] = True
    #             except:
    #                 print(f"Index Error: {a}, {b}, {idx}, {node_index}")
    #                 print(f"cfg_to_code: {self.examples[item].cfg_to_code}")
    #                 print(f"dfg_to_code: {self.examples[item].dfg_to_code}")
    #                 print(f"source_ids: {self.examples[item].source_ids}")
    #                 print(f"position_idx: {self.examples[item].position_idx}")
                    
    #     if len(self.examples[item].source_ids) != 320 or len(self.examples[item].source_mask) != 320 or len(self.examples[item].position_idx) != 320 or len(attn_mask) != 320:
    #         print(f"Source ID Length: {len(self.examples[item].source_ids)}")
    #         print(f"Source Mask Length: {len(self.examples[item].source_mask)}")
    #         print(f"Position Index Length: {len(self.examples[item].position_idx)}")
    #         print(f"Attention Mask Length: {len(attn_mask)}")
    #         print(f"Target ID Length: {len(self.examples[item].target_ids)}")
    #         print(f"Target Mask Length: {len(self.examples[item].target_mask)}")
    #     # return (torch.tensor(self.examples[item].source_ids),
    #     if len(self.examples[item].source_ids) != len(self.examples[item].source_mask):
    #         print(f"Source ID Length: {len(self.examples[item].source_ids)}")
    #         print(f"Source Mask Length: {len(self.examples[item].source_mask)}")
    #     return (torch.tensor(self.examples[item].source_ids),
    #             torch.tensor(self.examples[item].source_mask),
    #             torch.tensor(self.examples[item].position_idx),
    #             torch.tensor(attn_mask), 
    #             torch.tensor(self.examples[item].target_ids),
    #             torch.tensor(self.examples[item].target_mask),)
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  

    parser.add_argument("--source_lang", default=None, type=str, 
                        help="The language of input")  
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name )
    
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        train_data = TextDataset(train_features,args)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps,num_workers=54)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        
        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch
                loss,_,_ = model(source_ids,source_mask,position_idx,att_mask,target_ids,target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval and epoch in [ int(args.num_train_epochs*(i+1)//20) for i in range(20)]:
                                                                      #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    eval_data = TextDataset(eval_features,args)
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=54)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)               
                    source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch
                    with torch.no_grad():
                        _,loss,num = model(source_ids,source_mask,position_idx,att_mask,target_ids,target_mask)     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                     
                # Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   

                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model_ast.bin")
                torch.save(model_to_save.state_dict(), output_model_file)    
                                
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model_ast.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  


                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    eval_data = TextDataset(eval_features,args)
                    dev_dataset['dev_bleu']=eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=54)
                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch                 
                    with torch.no_grad():
                        preds = model(source_ids,source_mask,position_idx,att_mask)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                accs=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(ref)
                        f.write(ref+'\n')
                        f1.write(gold.target+'\n')     
                        accs.append(ref==gold.target)

                dev_bleu=round(_bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")),2)
                xmatch=round(np.mean(accs)*100,4)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
                logger.info("  "+"*"*20)    
                if dev_bleu+xmatch>best_bleu:
                    logger.info("  Best BLEU+xMatch:%s",dev_bleu+xmatch)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu+xmatch
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, f"pytorch_model_Best BLEU+xMatch_{dev_bleu+xmatch}.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               
    if args.do_test:
        files=[]
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            eval_data = TextDataset(eval_features,args) 

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=54)

            model.eval() 
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch                    
                with torch.no_grad():
                    preds = model(source_ids,source_mask,position_idx,att_mask)  
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions=[]
            accs=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(ref)
                    f.write(ref+'\n')
                    f1.write(gold.target+'\n')    
                    accs.append(ref==gold.target)
            dev_bleu=round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file), 
                                 os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)),2)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
            logger.info("  "+"*"*20)   
            
if __name__ == "__main__":
    main()

