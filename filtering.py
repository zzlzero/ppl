import nlqf
import os
import os.path as path
import json
import re
import torch
import javalang
import pickle
import numpy as np
import collections
from tables import *
from nltk import stem
from nltk.corpus import stopwords
from tqdm import tqdm
P = re.compile(r'([a-z]|\d)([A-Z])')
stemmer = stem.PorterStemmer()
code_stop_words = set(stopwords.words('codeStopWord'))
english_stop_words = set(stopwords.words('codeQueryStopWord'))
ka = re.compile(r'[^a-zA-Z]')

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

def extract_api_seq(code_str):
    code_str = "package nothing; class Hi {%s}" % code_str
    api_seq = []
    try:
        tree = javalang.parse.parse(code_str)
        identifier_filter = {}
        for _, node in tree:
            if isinstance(node, javalang.tree.FormalParameter):
                identifier_filter[node.name] = node.type.name

            if isinstance(node, javalang.tree.LocalVariableDeclaration):
                for dec in node.declarators:
                    identifier_filter[dec.name] = node.type.name

            if isinstance(node, javalang.tree.ClassCreator):
                api = [node.type.name, 'new']
                api_seq.append(api)
                api_seq.extend(check_selectors(node, identifier_filter))

            if isinstance(node, javalang.tree.MethodInvocation):
                if node.qualifier == '':
                    continue

                if node.qualifier is None:
                    if len(api_seq) == 0:
                        continue
                    node.qualifier = api_seq[-1][0]

                sub_api_seq = find_method(node, identifier_filter)
                sub_api_seq.append(
                    [identifier_filter.get(node.qualifier, node.qualifier),
                    node.member])
                api_seq.extend(sub_api_seq)
                api_seq.extend(check_selectors(node, identifier_filter))
    except:
        return ' '.join([' '.join(item) for item in api_seq])
    api_seq = [' '.join(item) for item in api_seq]
    return ' '.join(api_seq)

def extract_tokens(code_tokens):
    code_tokens = code2list(code_tokens) 
    tokens = list(set(code_tokens))
    return ' '.join(tokens)


def extract_method_name(camel_name):
    camel_name = camel_name.split('.')[-1]
    sub_name_tokens = split_camel(camel_name)
    sub_name_tokens = ' '.join(sub_name_tokens)
    return sub_name_tokens


def extract_docstring(docstring):
    if '.' in docstring:
        docstring = docstring.split('.')[0]
    elif '\n' in docstring:
        docstring = docstring.split('\n')[0]
    docstring = query2list(docstring)
    return ' '.join(docstring)

def find_method(method_node, ifilter):
    sub_api_seq = []
    for node in method_node.arguments:
        if isinstance(node, javalang.tree.MethodInvocation):
            api = [ifilter.get(node.qualifier, node.qualifier),
                   node.member]
            sub_api_seq.append(api)
            sub_api_seq.extend(find_method(node, ifilter))

        if isinstance(node, javalang.tree.ClassCreator):
            api = [node.type.name, 'new']
            sub_api_seq.append(api)
            sub_api_seq.extend(find_method(node, filter))
    return sub_api_seq


def check_selectors(node, s_filter):
    select_api_seq = []
    if node.selectors is not None:
        for sel in node.selectors:
            if isinstance(sel, javalang.tree.MethodInvocation):
                if node.qualifier is None:
                    select_api_seq.append([node.type.name, sel.member])
                else:
                    select_api_seq.append(
                        [s_filter.get(node.qualifier, node.qualifier),
                         sel.member])
    return select_api_seq


def generate_vocab(data):
    tokens = ' '.join(data).split(' ')
    count = collections.Counter(tokens)
    tokens_unique = ['<PAD>','<UNK>'] + [tok for tok, _ in count.most_common(9998)]
    vocab = {token: i for i, token in enumerate(tokens_unique)}
    return vocab

def code2list(raw):
    keep_alpha = re.sub(ka, ' ', raw)
    split_hump = re.sub(P, r'\1 \2', keep_alpha)
    lower = split_hump.lower().split()
    remove_stop = [w for w in lower if w not in code_stop_words]
    stemmed = [stemmer.stem(j) for j in remove_stop]
    return stemmed

def query2list(raw):
    keep_alpha = re.sub(ka, ' ', raw)
    split_hump = re.sub(P, r'\1 \2', keep_alpha)
    lower = split_hump.lower().split()
    remove_stop = [w for w in lower if w not in english_stop_words]
    stemmed = [stemmer.stem(j) for j in remove_stop]
    return stemmed


def token2id(data, vocab):
    id_list = []
    for token_str in data:
        tokens = token_str.split()
        ids = []
        for token in tokens:
            ids.append(vocab.get(token, 1))
        if len(ids) == 0:
            ids.append(0)
        id_list.append(ids)
    return id_list


def split_camel(camel_str):
    split_str = re.sub(
        r'(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+', '_',
        camel_str)
    return split_str.lower().split('_')


class Features(IsDescription):
    length = Int64Col()
    pos = Int64Col()


def save_features(data, name, dir_path):
    phrases = []
    pos_pointer = 0

    fea_f = open_file(os.path.join(dir_path, name), mode='w',
                      title='Data for deep code search')
    table = fea_f.create_table('/', 'indices', Features, "lalala")
    rows = table.row
    for item in data:
        phrases.extend(item)
        rows['length'] = len(item)
        rows['pos'] = pos_pointer
        pos_pointer += len(item)
        rows.append()
    table.flush()

    arr = fea_f.create_array('/', 'phrases', np.array(phrases))
    arr.flush()
    fea_f.close()


def save_vocab(vocab, name='', dir_path=''):
    os.makedirs(dir_path, exist_ok=True)
    f = open(os.path.join(dir_path, name), 'wb')
    pickle.dump(vocab, f)
    f.close()

def extract_features(doc, code_str, func_name, dir_path):
    api_seqs = []
    method_names = []
    tokens = []
    docstrings = []
    for d,c,f in zip(doc, code_str, func_name):
        docstring = extract_docstring(d)
        api_seq = extract_api_seq(c) 
        method_name = extract_method_name(f)
        token = extract_tokens(c)
        api_seqs.append(api_seq)
        method_names.append(method_name)
        tokens.append(token)
        docstrings.append(docstring)
    file_name = ['methname', 'apiseq', 'tokens', 'desc']
    vocabs = []
    for i, data_list in enumerate(
            [method_names, api_seqs, tokens, docstrings]):
        vocab = generate_vocab(data_list)
        save_vocab(vocab, 'vocab.{}.pkl'.format(file_name[i]), dir_path)
        vocabs.append(vocab)

    for j, data_list in enumerate(
            [method_names, api_seqs, tokens, docstrings]):
        train_id_list = token2id(data_list, vocabs[j])

        save_features(train_id_list, '{}.{}.h5'.format('train', file_name[j]),
                      dir_path)


from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
def caculate_ppl(raw_comments, raw_code):
    device = "cuda"
    #model_id = "gpt2-large"
    model_id = "/mnt/sda/zzl/NLQF/ppl/ft_r8_merged_data/epoch=20"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
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
  
    return perplexities
    '''    
    idx = [index for index, comment in enumerate(raw_comments) if comment in comments]
    comments = [raw_comments[i] for i in idx]
    code = [raw_code[i] for i in idx]
    ppl = [raw_ppl[i] for i in idx]
    df = pd.DataFrame({"句子": comments, "code": code, "困惑度": ppl})
    df.to_csv("perplexities_finetune15epoch_rulefilter.csv", index=False)
    '''
def model_filter(raw_comments, raw_code, with_cuda=True, query_len=20,num_workers=1,max_iter=1000,dividing_point=None):
    if not isinstance(raw_comments, list):
        raise TypeError('comments must be a list')
    
    if len(raw_comments) < 2:
        raise ValueError('The length of comments must be greater than 1')


    import pandas as pd
    index_list = [index for index, comment in enumerate(raw_comments) if comment != '']
    comments = [raw_comments[index] for index in index_list]
    print(type(comments))
    code = [raw_code[index] for index in index_list]
    print(type(code))
    
    
    # 使用pickle将comments和code保存到一个文件
    with open('comments_and_code.pkl', 'wb') as f:
        pickle.dump((comments, code), f)
    
    ppl = caculate_ppl(comments, code)

    # 保存ppl为文件
    df = pd.DataFrame({"sentence": comments, "code": code, "ppl": ppl})
    df.to_csv("perplexities_rulefilter.csv", index=False)
    
    # 读取ppl文件
    #with open('comments_and_code.pkl', 'rb') as f:
    #    comments, code = pickle.load(f)

    #df = pd.read_csv("perplexities_rulefilter.csv")
    #ppl = df["ppl"].tolist()    


    idx = np.argsort(ppl)
    top_80_percent_count = int(0.60 * len(idx))
    idx_new = idx[:top_80_percent_count]   
    filtered_comments = [comments[i] for i in idx_new] 
    filtered_code = [code[i] for i in idx_new]
    return filtered_comments, filtered_code


def filtering(raw_comments,raw_code,word_vocab,vae_model=None):
    #comments = raw_comments
    #code = raw_code
    comments, idx = nlqf.rule_filter(raw_comments)
    print('after rule filter:', len(comments))
    code = [raw_code[i] for i in idx]
    comments, code = model_filter(comments, code)
    print('after model filter:', len(comments))
    return comments, code

def extract_test_features(code, query_dir, save_dir):
    eval_queries = []
    with open(path.join(query_dir, 'query.jsonl'), 'r') as f:
        for row in f.readlines():
            eval_queries.append(json.loads(row))
    methname, apiseq, tokens = [], [], []
    for i in eval_queries:
        methname.append(extract_method_name(i['method_name']))
        apiseq.append(extract_api_seq(i['code']))
        tokens.append(extract_tokens(i['code']))

    for i in code: 
        methname.append(extract_method_name(i['func_name']))
        apiseq.append(extract_api_seq(i['original_string']))
        tokens.append(extract_tokens(i['original_string']))
    data_dic = { 
        'methname': methname,
        'apiseq': apiseq,
        'tokens': tokens,
    }
    for name,d in data_dic.items():
        with open(os.path.join(save_dir,f'vocab.{name}.pkl'),'rb') as f:
            vocab = pickle.load(f)
        id_list = token2id(d, vocab)
        save_features(id_list, '{}.{}.CSN.h5'.format('use', name), save_dir)
    with open(path.join(save_dir,'codebase_id.txt'), 'w') as f:
        f.write('\n'.join([str(i['id']) for i in eval_queries]+ [str(i['url']) for i in test_code]))

if __name__ == '__main__':
    CSN_train_path = './raw_dataset/java/train'
    CSN_test_path = './raw_dataset/java/test/'
    CSN_eval_path = './raw_dataset/java/eval/'
    vocab_path = './resource/word_vocab.json'
    #model_path = './resource/vae.model'
    save_dir = './processed_dataset'

    raw_comments, raw_code = load_CSN(CSN_train_path)
    print('train set size:', len(raw_comments))
    with open(vocab_path,'r') as f:
        word_vocab = json.load(f)
    #model = torch.load(model_path)
    queries, code = filtering(raw_comments,raw_code,word_vocab)
    print('filtered_size',len(queries))
    extract_features(queries, [co['original_string'] for co in code], [co['func_name'] for co in code], dir_path=save_dir)

    _, test_code = load_CSN(CSN_test_path)
    print(len(test_code))
    extract_test_features(test_code, CSN_eval_path, save_dir)



