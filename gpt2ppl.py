from calendar import c
from numpy import mean
from sympy import per
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
device = "cuda"
#model_id = "gpt2-large"
model_id = "exp/epoch=20"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
import torch
from tqdm import tqdm
from datasets import load_dataset
# test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
import os
from os import path
import json
import re
def get_first_sentence(docstring): 
    docstring = re.split(r'[.\n\r]',docstring.strip('\n'))[0]
    return docstring
def load_CSN(CSN_path):
    raw_comments = []
    raw_code = []
    for file in os.listdir(CSN_path):
        with open(path.join(CSN_path,file), 'r') as f:
            for row in f.readlines():
                row_obj = json.loads(row)
                raw_comments.append(get_first_sentence(row_obj['docstring']))
                raw_code.append({
                    'func_name': row_obj['func_name'].split('.')[0],
                    'original_string': row_obj['original_string'],
                    'url': row_obj['url']
            })
    return raw_comments, raw_code

CSN_train_path = './raw_dataset/java/train/'
raw_comments, raw_code = load_CSN(CSN_train_path)
index_list = [index for index, comment in enumerate(raw_comments) if comment != '']
raw_comments = [raw_comments[index] for index in index_list]
raw_code = [raw_code[index] for index in index_list]
print('' in raw_comments)
print((raw_comments[:35]))
'''
encodings = tokenizer("\n\n".join(raw_comments), return_tensors="pt")


max_length = model.config.n_positions
stride = 1024
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item)
'''

def caculate_ppl(raw_comments, raw_code):
    perplexities = []
    max_length = model.config.n_positions
    stride = 1024
    # 逐句计算困惑度
    for comment in tqdm(raw_comments, desc="计算困惑度", unit="句"):
        # 将句子进行tokenization，并将结果转换为PyTorch张量
        encoding = tokenizer(comment, return_tensors="pt").to(device)
        seq_len = encoding.input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encoding.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        # 计算困惑度，并添加到列表中
        ppl = torch.exp(torch.stack(nlls).mean())
        # print(ppl.item)
        perplexities.append(ppl.item())
    # 打印每个句子的困惑度
    for i, perplexity in enumerate(perplexities[:100]):
        print(f"句子 {i+1}\t困惑度 = {perplexity}\t{raw_comments[i]}")
    # 输出到表格
    table = pd.DataFrame({"sentence": raw_comments, "code": raw_code, "ppl": perplexities})
    table.to_csv("perplexities_finetune20epoch.csv", index=False)
    return perplexities

ppl_list = caculate_ppl(raw_comments, raw_code)
print(mean(ppl_list))

