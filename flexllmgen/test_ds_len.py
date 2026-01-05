import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from collections import defaultdict
from datasets import load_dataset

def process_full_dapr():
    RootPath = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/datasets/"
    docs = load_dataset(RootPath + "UKPLab___dapr/ConditionalQA-docs/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/", split="test")
    qrels = load_dataset(RootPath + "UKPLab___dapr/ConditionalQA-qrels/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/", split="test")
    queries = load_dataset(RootPath + "UKPLab___dapr/ConditionalQA-queries/0.0.0/67ae3daa13596700976d20605630f5f9db3bd732/", split="test")
    print(queries)
    qrels_dict = defaultdict(set)

    for row in qrels:
        corpus_id = row["corpus_id"]
        doc_id = corpus_id.split('-')[0]
        qrels_dict[doc_id].add(row["query_id"])
    
    def process_doc(docs, doc_id):
        docs = docs.filter(lambda row: row["doc_id"] == doc_id)
        passages = docs[0]["passages"]
        context = ""
        for psg in passages:
            context += psg + "\n"
        return context
    def process_query(queries, query_ids):
        questions = []
        target_queries = queries.filter(lambda q: q["_id"] in query_ids)
        for q in target_queries:
            questions.append(q["text"])
        return questions

    process_res = {}
    for k, v in qrels_dict.items():
        if len(v) < 5:
            continue
        doc_id, query_ids = k, v
        context = process_doc(docs, doc_id)
        questions = process_query(queries, query_ids)
        process_res[doc_id] = (context, questions)

    return process_res


def process_full_longbench():
    RootPath = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/datasets/THUDM___long_bench/data/"
    task = "narrativeqa"
    file_path = RootPath + task + ".jsonl"
    # print(file_path)
    dataset = load_dataset('json', data_files=file_path)["train"]
    
    context_to_questions = defaultdict(list)
    
    for row in dataset:
        context = row["context"]
        question = row["input"]
        context_to_questions[context].append(question)
    
    requests = {}
    req_id = 0
    for ctx, qs in context_to_questions.items():
        if (len(qs)) < 10:
            continue
        max_context = ctx[:18000]
        requests[req_id] = (max_context, qs)
        req_id += 1

    return requests


tokenizer = AutoTokenizer.from_pretrained("/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/param/opt-30b", truncation_side="left")
dapr = process_full_dapr()
dapr_inputs = []
for doc_id, (context, questions) in dapr.items():
    inputs = [context +  query + "\n" for query in questions]
    inputs_ids = tokenizer(inputs, truncation=True, max_length=5000).input_ids
    for inputid in inputs_ids:
        dapr_inputs.append(len(inputid))
avg_len = sum(dapr_inputs) / len(dapr_inputs)
print(f"{len(dapr_inputs)} reqs in dapr, avg len: {avg_len}")

longbench = process_full_longbench()
longbench_inputs = []
for doc_id, (context, questions) in longbench.items():
    inputs = [context +  query + "\n" for query in questions]
    inputs_ids = tokenizer(inputs, truncation=True, max_length=7000).input_ids
    for inputid in inputs_ids:
        longbench_inputs.append(len(inputid))
avg_len = sum(longbench_inputs) / len(longbench_inputs)
print(f"{len(longbench_inputs)} reqs in longbench, avg len: {avg_len}")