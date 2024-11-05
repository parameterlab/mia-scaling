# %%
import logging

logging.basicConfig(level='ERROR')
import argparse
import json
import os
import random
import zlib

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
This script uses the first paragraph of 2048 tokens from each document in the collection to compute the MIA scores.
'''

# %%
def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    encodings = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=2048)
    if model.device.type == "cuda":
        encodings = {k: v.cuda() for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = encodings['input_ids'][0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


# %%
# in run.py you have a variant of this function with one more MIA: ppl/Ref_ppl
def inference(model1, tokenizer1, text):
    pred = {}

    p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1)
    p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1)


# ppl
    pred["ppl"] = p1

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1)/zlib_entropy
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    return pred
# %%
def create_text(x):
    conversation = x['conversations']
    text = ""
    for message in conversation:
        text += message['from'] + ": " + message['value'] + "\n"
    return {"text": text}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--dataset_name", type=str, default="haritzpuerto/the_pile_00_arxiv")
    parser.add_argument("--filter_outliers", action="store_true")
    parser.add_argument("--min_chars", type=int, default=100)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="/tmp")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    # %%
    model1 = AutoModelForCausalLM.from_pretrained(args.model_name, return_dict=True, device_map='auto', torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)
    model1.eval()
    tokenizer1 = AutoTokenizer.from_pretrained(args.model_name)

    # %%
    ds = load_dataset(args.dataset_name)
    if args.filter_outliers:
        # removing outlier docs
        ds['train'] = ds['train'].filter(lambda x: len(x["text"]) > args.min_chars)
        ds['validation'] = ds['validation'].filter(lambda x: len(x["text"]) > args.min_chars)
        ds['test'] = ds['test'].filter(lambda x: len(x["text"]) > args.min_chars)

    nonmembers = concatenate_datasets([ds["validation"], ds["test"]])
    members = ds["train"].shuffle(seed=args.seed).select(range(len(nonmembers)))

    nonmembers.save_to_disk(os.path.join(args.output_path, "nonmembers"))
    members.save_to_disk(os.path.join(args.output_path, "members"))
    # %%
    data_points_members = []
    for text in tqdm(members['text']):
        mia_features = inference(model1, tokenizer1, text)
        data_points_members.append({'pred': mia_features, 'label': 1})
        torch.cuda.empty_cache()

    with open(os.path.join(args.output_path, "mia_members.jsonl"), "w") as f:
        for dp in data_points_members:
            f.write(json.dumps(dp) + "\n")

    data_points_nonmembers = []
    for text in tqdm(nonmembers['text']):
        mia_features = inference(model1, tokenizer1, text)
        data_points_nonmembers.append({'pred': mia_features, 'label': 0})
        torch.cuda.empty_cache()

    with open(os.path.join(args.output_path, "mia_nonmembers.jsonl"), "w") as f:
        for dp in data_points_nonmembers:
            f.write(json.dumps(dp) + "\n")
