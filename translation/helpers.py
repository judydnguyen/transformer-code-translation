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
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
import warnings
warnings.filterwarnings("ignore")
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


def extract_controlflow(code, parser, lang):
    """
    Extract control flow edges (CFG) from the code.
    """
    try:
        # Parse the code to obtain the AST
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node

        # Extract control flow edges from the AST
        cfg = extract_control_flow_edges(root_node)
        return cfg
    except Exception as e:
        print(f"Error extracting CFG: {e}")
        return []


def extract_control_flow_edges(root_node):
    """
    Extract control flow edges (e.g., branching, loops) from the AST.
    """
    edges = []

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
                edges.append((condition_node.start_point, then_node.start_point))
            # Extract the "else" block, if present
            else_node = next(
                (child for child in node.children if child.type == "else_body"), None
            )
            if condition_node and else_node:
                edges.append((condition_node.start_point, else_node.start_point))
        elif node.type in ("for_statement", "while_statement"):
            # Extract loop condition and body
            condition_node = next(
                (child for child in node.children if child.type == "parenthesized_expression"), None
            )
            body_node = next(
                (child for child in node.children if child.type == "block"), None
            )
            if condition_node and body_node:
                edges.append((condition_node.start_point, body_node.start_point))
        elif node.type == "return_statement":
            # Add a return statement as a flow edge
            edges.append((node.start_point, node.end_point))

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

def augment_code(code_sample, lang):
    """
    Augment the code sample by adding control flow and data flow edges.
    """
    parser = parsers[lang]
    code_tokens, dfg = extract_dataflow(code_sample, parser, lang)
    cfg = extract_controlflow(code_sample, parser, lang)
    return code_tokens, cfg, dfg

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

import random
import string

def replace_variable_names_with_dfg_randomized_strings(code_sample, lang, parser):
    """
    Replace variable names in the code using information from the Data Flow Graph (DFG) with randomized string suffixes.
    Args:
        code_sample (str): The source code to augment.
        lang (str): Programming language (e.g., 'java', 'c_sharp').
        parser: Parser for extracting DFG.
    Returns:
        str: Code with replaced variable names.
    """
    # Extract tokens and dataflow
    code_tokens, dfg = extract_dataflow(code_sample, parser, lang)

    # Extract variable names from DFG
    variable_names = [df[0] for df in dfg if df[0].isidentifier()]  # Filter variable names
    if not variable_names:
        return code_sample  # No variables to replace

    # Create a replacement map with randomized string suffixes of length between 4 and 10
    used_suffixes = set()  # To ensure unique variable names
    replacement_map = {}
    for var in variable_names:
        while True:  # Ensure no duplicate random suffixes
            random_suffix = ''.join(random.choices(string.ascii_letters, k=random.randint(4, 10)))
            if random_suffix not in used_suffixes:
                used_suffixes.add(random_suffix)
                break
        replacement_map[var] = f"{random_suffix}"

    # Replace variable names in the tokens
    replaced_tokens = [replacement_map.get(token, token) for token in code_tokens]

    # Reconstruct the code
    replaced_code = " ".join(replaced_tokens)
    return replaced_code


def replace_variable_names_with_dfg(code_sample, lang, parser):
    """
    Replace variable names in the code using information from the Data Flow Graph (DFG).
    Args:
        code_sample (str): The source code to augment.
        lang (str): Programming language (e.g., 'java', 'c_sharp').
        parser: Parser for extracting DFG.
    Returns:
        str: Code with replaced variable names.
    """
    # Extract tokens and dataflow
    code_tokens, dfg = extract_dataflow(code_sample, parser, lang)

    # Extract variable names from DFG
    variable_names = [df[0] for df in dfg if df[0].isidentifier()]  # Filter variable names
    if not variable_names:
        return code_sample  # No variables to replace

    # Create a replacement map for variables
    # random a new variable name in range [4,10]
    
    replacement_map = {var: f"var_{i}" for i, var in enumerate(variable_names)}

    # Replace variable names in the tokens
    replaced_tokens = [replacement_map.get(token, token) for token in code_tokens]

    # Reconstruct the code
    replaced_code = " ".join(replaced_tokens)
    return replaced_code

    
def generate_augmented_versions(code_sample, lang, parser, num_versions=5, delete_prob=0.1, swap_prob=0.1):
    """
    Generate multiple augmented versions of a code sample by randomly activating delete or swap functions.
    Args:
        code_sample (str): The source code to augment.
        lang (str): Programming language (e.g., 'java', 'c_sharp').
        parser: Parser for extracting DFG.
        num_versions (int): Number of augmented versions to generate.
        delete_prob (float): Probability of deleting a token.
        swap_prob (float): Probability of swapping two tokens.
    Returns:
        list: List of augmented code versions.
    """
    # Extract tokens and dataflow
    code_tokens, dfg = extract_dataflow(code_sample, parser, lang)
    
    # Random deletion function
    def random_delete(tokens, prob):
        return [t for t in tokens if random.random() > prob]

    # Random swap function
    def random_swap(tokens, prob):
        if len(tokens) < 2:
            return tokens  # Nothing to swap
        new_tokens = tokens[:]
        for _ in range(int(len(tokens) * prob)):
            i, j = random.sample(range(len(tokens)), 2)
            new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
        return new_tokens

    augmented_versions = []
    for _ in range(num_versions):
        tokens = code_tokens[:]
        
        # Randomly activate delete or swap
        if random.choice([True, False]):  # Randomly decide to activate delete
            tokens = random_delete(tokens, delete_prob)
        if random.choice([True, False]):  # Randomly decide to activate swap
            tokens = random_swap(tokens, swap_prob)
        
        # Reconstruct the code
        augmented_code = " ".join(tokens)
        augmented_versions.append(augmented_code)

    return augmented_versions

def generate_augmented_protected_versions(code_sample, lang, parser, num_versions=5, delete_prob=0.1, swap_prob=0.1):
    """
    Generate multiple augmented versions of a code sample by randomly activating delete or swap functions,
    while protecting control flow keywords.
    Args:
        code_sample (str): The source code to augment.
        lang (str): Programming language (e.g., 'java', 'c_sharp').
        parser: Parser for extracting DFG.
        num_versions (int): Number of augmented versions to generate.
        delete_prob (float): Probability of deleting a token.
        swap_prob (float): Probability of swapping two tokens.
    Returns:
        list: List of augmented code versions.
    """
    # Extract tokens and dataflow
    code_tokens, dfg = extract_dataflow(code_sample, parser, lang)

    # Define protected keywords (e.g., control flow keywords)
    protected_keywords = {"if", "else", "for", "while", "return", "switch", "case", "try", "catch", "finally", "break", "continue"}

    # Helper function to check if a token is protected
    def is_protected(token):
        return token in protected_keywords

    # Random deletion function
    def random_delete(tokens, prob):
        return [t for t in tokens if random.random() > prob or is_protected(t)]

    # Random swap function
    def random_swap(tokens, prob):
        if len(tokens) < 2:
            return tokens  # Nothing to swap
        new_tokens = tokens[:]
        for _ in range(int(len(tokens) * prob)):
            # Find two non-protected indices to swap
            swap_candidates = [i for i, t in enumerate(new_tokens) if not is_protected(t)]
            if len(swap_candidates) < 2:
                break  # Not enough tokens to swap
            i, j = random.sample(swap_candidates, 2)
            new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
        return new_tokens

    augmented_versions = []
    for _ in range(num_versions):
        tokens = code_tokens[:]
        
        # Randomly activate delete or swap
        if random.choice([True, False]):  # Randomly decide to activate delete
            tokens = random_delete(tokens, delete_prob)
        if random.choice([True, False]):  # Randomly decide to activate swap
            tokens = random_swap(tokens, swap_prob)
        
        # Reconstruct the code
        augmented_code = " ".join(tokens)
        augmented_versions.append(augmented_code)

    return augmented_versions

def augment_source_and_target(code_sample, target_sample, lang, parser, num_versions=5, delete_prob=0.1, swap_prob=0.1):
    """
    Generate augmented versions of source and target code samples.
    Args:
        code_sample (str): The source code to augment.
        target_sample (str): The target code to augment.
        lang (str): Programming language (e.g., 'java', 'c_sharp').
        parser: Parser for extracting DFG.
        num_versions (int): Number of augmented versions to generate.
        delete_prob (float): Probability of deleting a token.
        swap_prob (float): Probability of swapping two tokens.
    Returns:
        list: List of tuples (augmented_source, augmented_target).
    """
    # Extract tokens and dataflow for source
    source_tokens, source_dfg = extract_dataflow(code_sample, parser, lang)

    # Extract tokens and dataflow for target
    target_tokens, target_dfg = extract_dataflow(target_sample, parser, lang)

    # Define protected keywords
    protected_keywords = {"if", "else", "for", "while", "return", "switch", "case", "try", "catch", "finally", "break", "continue"}

    def is_protected(token):
        return token in protected_keywords

    def random_delete(tokens, prob):
        return [t for t in tokens if random.random() > prob or is_protected(t)]

    def random_swap(tokens, prob):
        if len(tokens) < 2:
            return tokens
        new_tokens = tokens[:]
        for _ in range(int(len(tokens) * prob)):
            swap_candidates = [i for i, t in enumerate(new_tokens) if not is_protected(t)]
            if len(swap_candidates) < 2:
                break
            i, j = random.sample(swap_candidates, 2)
            new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
        return new_tokens

    # Generate augmented source and target versions
    augmented_pairs = []
    for _ in range(num_versions):
        source_aug = source_tokens[:]
        target_aug = target_tokens[:]

        # Apply random deletions and swaps consistently
        if random.choice([True, False]):
            source_aug = random_delete(source_aug, delete_prob)
            target_aug = random_delete(target_aug, delete_prob)
        if random.choice([True, False]):
            source_aug = random_swap(source_aug, swap_prob)
            target_aug = random_swap(target_aug, swap_prob)

        # Reconstruct augmented source and target
        augmented_source = " ".join(source_aug)
        augmented_target = " ".join(target_aug)
        augmented_pairs.append((augmented_source, augmented_target))

    return augmented_pairs

def generate_augmented_versions_with_constraints(
    code_sample, lang, parser, num_versions=5, delete_prob=0.1, swap_prob=0.1, max_deletions=3, max_swaps=2
):
    """
    Generate augmented versions of a code sample with constraints on deletions and swaps.
    Args:
        code_sample (str): The source code to augment.
        lang (str): Programming language (e.g., 'java', 'c_sharp').
        parser: Parser for extracting DFG.
        num_versions (int): Number of augmented versions to generate.
        delete_prob (float): Probability of deleting a token.
        swap_prob (float): Probability of swapping two tokens.
        max_deletions (int): Maximum number of tokens to delete.
        max_swaps (int): Maximum number of swaps to perform.
    Returns:
        list: List of augmented code versions.
    """
    # Extract tokens and dataflow
    code_tokens, dfg = extract_dataflow(code_sample, parser, lang)

    # Random deletion function with constraints
    def random_delete(tokens, prob, max_deletions):
        deletions = 0
        result = []
        for token in tokens:
            if deletions < max_deletions and random.random() < prob:
                deletions += 1  # Count this deletion
            else:
                result.append(token)
        return result

    # Random swap function with constraints
    def random_swap(tokens, prob, max_swaps):
        if len(tokens) < 2 or max_swaps <= 0:
            return tokens  # Nothing to swap
        new_tokens = tokens[:]
        swaps = 0
        while swaps < max_swaps:
            if random.random() >= prob:
                break  # Stop if probability condition not met
            i, j = random.sample(range(len(new_tokens)), 2)
            new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
            swaps += 1  # Increment swap count
        return new_tokens

    augmented_versions = []
    for _ in range(num_versions):
        tokens = code_tokens[:]

        # Apply constrained random deletions and swaps
        if random.choice([True, False]):  # Randomly decide to activate delete
            tokens = random_delete(tokens, delete_prob, max_deletions)
        if random.choice([True, False]):  # Randomly decide to activate swap
            tokens = random_swap(tokens, swap_prob, max_swaps)

        # Reconstruct the code
        augmented_code = " ".join(tokens)
        augmented_versions.append(augmented_code)

    return augmented_versions

def augment_pair_with_constraints(
    source_code,
    target_code,
    lang,
    parser,
    num_versions=5,
    delete_prob=0.1,
    swap_prob=0.1,
    max_deletions=3,
    max_swaps=2
):
    """
    Augment a pair of source and target code samples with constraints on deletions and swaps.
    Args:
        source_code (str): The source code to augment.
        target_code (str): The target code to augment.
        lang (str): Programming language (e.g., 'java', 'c_sharp').
        parser: Parser for extracting DFG.
        num_versions (int): Number of augmented versions to generate.
        delete_prob (float): Probability of deleting a token.
        swap_prob (float): Probability of swapping two tokens.
        max_deletions (int): Maximum number of tokens to delete.
        max_swaps (int): Maximum number of swaps to perform.
    Returns:
        list: List of tuples (augmented_source, augmented_target).
    """
    # Extract tokens and DFG for source and target
    source_tokens, source_dfg = extract_dataflow(source_code, parser, lang)
    target_tokens, target_dfg = extract_dataflow(target_code, parser, lang)

    # Ensure consistent variable renaming using DFG
    variable_names = [df[0] for df in source_dfg if df[0].isidentifier()]
    replacement_map = {
        var: f"var_{i}" for i, var in enumerate(variable_names)
    }  # Consistent renaming map

    # Replace variable names in source and target
    def replace_variables(tokens, dfg, replacement_map):
        return [
            replacement_map.get(token, token) for token in tokens
        ]  # Replace variables using the map

    source_tokens = replace_variables(source_tokens, source_dfg, replacement_map)
    target_tokens = replace_variables(target_tokens, target_dfg, replacement_map)

    # Augmentation functions
    def random_delete(tokens, prob, max_deletions):
        deletions = 0
        result = []
        for token in tokens:
            if deletions < max_deletions and random.random() < prob:
                deletions += 1  # Count this deletion
            else:
                result.append(token)
        return result

    def random_swap(tokens, prob, max_swaps):
        if len(tokens) < 2 or max_swaps <= 0:
            return tokens  # Nothing to swap
        new_tokens = tokens[:]
        swaps = 0
        while swaps < max_swaps:
            if random.random() >= prob:
                break  # Stop if probability condition not met
            i, j = random.sample(range(len(new_tokens)), 2)
            new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
            swaps += 1  # Increment swap count
        return new_tokens

    # Generate augmented pairs
    augmented_pairs = []
    for _ in range(num_versions):
        src_aug = source_tokens[:]
        tgt_aug = target_tokens[:]

        # Randomly apply deletion and swap
        if random.choice([True, False]):  # Apply delete
            src_aug = random_delete(src_aug, delete_prob, max_deletions)
            tgt_aug = random_delete(tgt_aug, delete_prob, max_deletions)
        if random.choice([True, False]):  # Apply swap
            src_aug = random_swap(src_aug, swap_prob, max_swaps)
            tgt_aug = random_swap(tgt_aug, swap_prob, max_swaps)

        # Reconstruct code
        augmented_source = " ".join(src_aug)
        augmented_target = " ".join(tgt_aug)
        augmented_pairs.append((augmented_source, augmented_target))

    return augmented_pairs

def augment_pair_with_dfg_validation(
    source_code,
    target_code,
    lang,
    parser,
    num_versions=5,
    delete_prob=0.1,
    swap_prob=0.1,
    max_deletions=1,
    max_swaps=1
):
    """
    Augment a pair of source and target code samples with constraints on deletions and swaps,
    ensuring DFG validation. Brackets and structural tokens are protected from deletion.
    Args:
        source_code (str): The source code to augment.
        target_code (str): The target code to augment.
        lang (str): Programming language (e.g., 'java', 'c_sharp').
        parser: Parser for extracting DFG.
        num_versions (int): Number of valid augmented pairs to generate.
        delete_prob (float): Probability of deleting a token.
        swap_prob (float): Probability of swapping two tokens.
        max_deletions (int): Maximum number of tokens to delete.
        max_swaps (int): Maximum number of swaps to perform.
    Returns:
        list: List of validated tuples (augmented_source, augmented_target).
    """
    def validate_dfg_consistency(source_code, target_code):
        """
        Validate if the DFGs of the source and target code are consistent.
        """
        _, source_dfg = extract_dataflow(source_code, parser, lang)
        _, target_dfg = extract_dataflow(target_code, parser, lang)

        # Extract variable names and their dependencies
        source_vars = {df[0]: set(df[-1]) for df in source_dfg if df[0].isidentifier()}
        target_vars = {df[0]: set(df[-1]) for df in target_dfg if df[0].isidentifier()}

        # Check if all source variables are present in the target with matching dependencies
        for var, dependencies in source_vars.items():
            if var not in target_vars or target_vars[var] != dependencies:
                return False  # Semantic mismatch
        return True

    # Extract tokens and DFG for source and target
    source_tokens, source_dfg = extract_dataflow(source_code, parser, lang)
    target_tokens, target_dfg = extract_dataflow(target_code, parser, lang)

    # Ensure consistent variable renaming using DFG
    variable_names = [df[0] for df in source_dfg if df[0].isidentifier()]
    # replacement_map = {
    #     var: f"var_{i}" for i, var in enumerate(variable_names)
    # }  # Consistent renaming map
    # Create a replacement map with randomized string suffixes of length between 4 and 10
    used_suffixes = set()  # To ensure unique variable names
    replacement_map = {}
    for var in variable_names:
        while True:  # Ensure no duplicate random suffixes
            random_suffix = ''.join(random.choices(string.ascii_letters, k=random.randint(4, 10)))
            if random_suffix not in used_suffixes:
                used_suffixes.add(random_suffix)
                break
        replacement_map[var] = f"{random_suffix}"

    # Replace variable names in source and target
    def replace_variables(tokens, dfg, replacement_map):
        return [
            replacement_map.get(token, token) for token in tokens
        ]  # Replace variables using the map

    source_tokens = replace_variables(source_tokens, source_dfg, replacement_map)
    target_tokens = replace_variables(target_tokens, target_dfg, replacement_map)

    # Augmentation functions
    def random_delete(tokens, prob, max_deletions):
        """
        Randomly delete tokens with a given probability, ensuring brackets and structural tokens are protected.
        """
        deletions = 0
        protected_tokens = {"{", "}", "(", ")", "[", "]"}  # Protect these tokens
        result = []
        for token in tokens:
            if (
                deletions < max_deletions
                and random.random() < prob
                and token not in protected_tokens
            ):
                deletions += 1  # Count this deletion
            else:
                result.append(token)
        return result

    def random_swap_nearest(tokens, prob, max_swaps):
        """
        Swap tokens only between nearest neighbors.
        """
        if len(tokens) < 2 or max_swaps <= 0:
            return tokens  # Nothing to swap
        new_tokens = tokens[:]
        swaps = 0
        while swaps < max_swaps:
            if random.random() >= prob:
                break  # Stop if probability condition not met
            i = random.randint(0, len(new_tokens) - 2)  # Choose a position to swap with its neighbor
            new_tokens[i], new_tokens[i + 1] = new_tokens[i + 1], new_tokens[i]
            swaps += 1  # Increment swap count
        return new_tokens

    # Generate augmented pairs
    augmented_pairs = []
    attempts = 0  # Track attempts to avoid infinite loops
    while len(augmented_pairs) < num_versions and attempts < num_versions * 1000:
        attempts += 1
        src_aug = source_tokens[:]
        tgt_aug = target_tokens[:]

        # Randomly apply delete and nearest-neighbor swap
        if random.choice([True, False]):  # Apply delete
            src_aug = random_delete(src_aug, delete_prob, max_deletions)
            tgt_aug = random_delete(tgt_aug, delete_prob, max_deletions)
        if random.choice([True, False]):  # Apply nearest-neighbor swap
            src_aug = random_swap_nearest(src_aug, swap_prob, max_swaps)
            tgt_aug = random_swap_nearest(tgt_aug, swap_prob, max_swaps)

        # Reconstruct code
        augmented_source = " ".join(src_aug)
        augmented_target = " ".join(tgt_aug)

        # Validate DFG consistency
        if validate_dfg_consistency(augmented_source, augmented_target):
            augmented_pairs.append((augmented_source, augmented_target))

    return augmented_pairs


if __name__ == '__main__':
    # Set seed
    set_seed(42)
    
    # make dir if output_dir not exist
    model_type = 'roberta'
    config_name = 'microsoft/graphcodebert-base'
    tokenizer_name = 'microsoft/graphcodebert-base'
    
    custom_file = "custom_data/valid.source.txt.java,custom_data/valid.target.txt.cs"
    original_file = f"data/train.java-cs.txt.java,data/train.java-cs.txt.cs"
    
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(config_name, force_download=True)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, force_download=True)
    
    # custom_eval_examples = read_examples(custom_file)  
    original_eval_examples = read_examples(original_file) 

    # loop over the examples
    augmented_src_file = open("custom_data/valid.aug.source.txt.java", "a+")
    augmented_tgt_file = open("custom_data/valid.aug.target.txt.cs", "a+")
    
    for example in original_eval_examples:
        # Extract source and target code samples
        source_code_sample = example.source
        target_code_sample = example.target
        lang = example.lang
        parser = parsers[lang]

        # Augment the source code
        augmented_source = replace_variable_names_with_dfg_randomized_strings(source_code_sample, lang, parser)
        print(f"\nAugmented Source:\n{augmented_source}")

        # Augment the target code
        augmented_target = replace_variable_names_with_dfg_randomized_strings(target_code_sample, lang, parser)
        print(f"\nAugmented Target:\n{augmented_target}")

        # Augment the source and target code samples
        augmented_pairs = augment_pair_with_dfg_validation(
            source_code_sample,
            target_code_sample,
            lang,
            parser,
            num_versions=5,
            delete_prob=0.1,
            swap_prob=0.1,
            max_deletions=2,
            max_swaps=1
        )
        # save the augmented source and target
        for src, tgt in augmented_pairs:
            # convert into a single line
            src = src.replace("\n", " ")
            tgt = tgt.replace("\n", " ")
            augmented_src_file.write(src + "\n")
            augmented_tgt_file.write(tgt + "\n")
    augmented_src_file.close()
    augmented_tgt_file.close()