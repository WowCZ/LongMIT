'''
Description: 
Version: 1.0
Autor: Zhi Chen
Date: 2024-07-18 03:34:19
LastEditors: chenzhi chenzhi@pjlab.org.cn
LastEditTime: 2024-08-07 04:54:09
'''
import os, sys
sys.path.append(os.getcwd())

import re
import json
import math
import copy
import random
import subprocess
from tqdm import tqdm
from glob import glob
from typing import List, Union
from multiprocessing import Pool
from doc_process.utils import read_jsonl_file


def recover_multihop_info(jsonl_file: str, saved_dir: str):
    print(jsonl_file)
    fname = jsonl_file.split('/')[-1]
    save_file = os.path.join(saved_dir, fname)

    statistics = {
        'intra': {},
        'inter': {}
    }

    with open(save_file, 'w') as fw:
        with read_jsonl_file(jsonl_file) as doc_reader:
            for path_id, doc_path in enumerate(doc_reader):
                all_docs = []
                for doc in doc_path['aggregated_docs']:
                    all_docs.append(doc)

                all_doc_qas = []
                for doc_qas in doc_path['intra_doc_qas']:
                    chunk_qas = []
                    for qas in doc_qas:
                        chunk_qas.extend(qas)
                    all_doc_qas.append(chunk_qas)
                
                assert len(all_docs) == len(all_doc_qas)

                single_qas = []
                reserved_docs = []
                qa_doc_map = {}
                for doc_qas, doc in zip(all_doc_qas, all_docs):
                    if len(doc_qas) > 0:
                        qi = len(single_qas)
                        di = len(reserved_docs)
                        for i in range(len(doc_qas)):
                            qa_doc_map[qi+i] = di
                            
                        single_qas.extend(doc_qas)
                        reserved_docs.append(doc)

                selected_qa_ids = sorted(doc_path['selected_qa_ids'])
                multihop_qas = doc_path['multihop_qas']

                clustered_mqa_info = doc_path['clustered_mqa_info']
                assert len(multihop_qas) == len(selected_qa_ids)

                for si, mqa in zip(selected_qa_ids, multihop_qas):
                    mqa_ids = clustered_mqa_info[si]
                    indoc_mqa_ids = mqa_ids['indoc_ids']
                    crossdoc_mqa_ids = mqa_ids['crossdoc_ids']

                    indoc_mqas = mqa['indoc']
                    crossdoc_mqas = mqa['crossdoc']

                    if len(indoc_mqas) > 0:
                        qa1 = single_qas[si]
                        d1i = qa_doc_map[si]
                        d1 = reserved_docs[d1i]
                        clue_qas = [qa1]
                        clue_docs = [d1]
                        clue_doc_ids = [d1i]
                        for qi in indoc_mqa_ids:
                            qa2 = single_qas[qi]
                            d2i = qa_doc_map[qi]
                            d2 = reserved_docs[d2i]
                            clue_qas.append(qa2)
                            clue_docs.append(d2)
                            clue_doc_ids.append(d2i)

                        other_docs = [{'id': d['id'], 'content': d['content']} for di, d in enumerate(reserved_docs) if di not in clue_doc_ids]
                        for i, imqa in enumerate(indoc_mqas):
                            recover_mq = {
                                'path_id': path_id,
                                'type': 'intra_doc',
                                'hop': i+2,
                                'clue_docs': clue_docs,
                                'clue_qas': clue_qas,
                                'mqa': imqa,
                                'other_docs': other_docs
                            }
                            fw.write(json.dumps(recover_mq, ensure_ascii=False)+'\n')

                            statistics['intra'][i+2] = statistics['intra'].get(i+2, 0) + 1

                    if len(crossdoc_mqas) > 0:
                        qa1 = single_qas[si]
                        d1i = qa_doc_map[si]
                        d1 = reserved_docs[d1i]
                        clue_qas = [qa1]
                        clue_docs = [d1]
                        clue_doc_ids = [d1i]
                        for qi in crossdoc_mqa_ids:
                            qa2 = single_qas[qi]
                            d2i = qa_doc_map[qi]
                            d2 = reserved_docs[d2i]
                            clue_qas.append(qa2)
                            clue_docs.append(d2)
                            clue_doc_ids.append(d2i)
                        
                        other_docs = [{'id': d['id'], 'content': d['content']} for di, d in enumerate(reserved_docs) if di not in clue_doc_ids]
                        for i, imqa in enumerate(crossdoc_mqas):
                            recover_mq = {
                                'path_id': path_id,
                                'type': 'inter_doc',
                                'hop': i+2,
                                'clue_docs': clue_docs,
                                'clue_qas': clue_qas,
                                'mqa': imqa,
                                'other_docs': other_docs
                            }
                            fw.write(json.dumps(recover_mq, ensure_ascii=False)+'\n')

                            statistics['inter'][i+2] = statistics['inter'].get(i+2, 0) + 1
    print(statistics)

def sample_doc(jsonl_files: List[str], sample_num: int, select_ids: List[tuple]=[], chosen_num: int=1):
    def wc_jsonl_lines(jsonl_file):
        result = subprocess.run(['wc', '-l', jsonl_file], capture_output=True, text=True)
        if result.returncode == 0:
            line_count = int(result.stdout.split()[0])
            # print(f"File {jsonl_file} has {line_count} lines.")
        else:
            print(f"Error in processing file {jsonl_file}: {result.stderr}")
        return line_count

    sampled_ids = []
    if len(select_ids) == 0:
        while len(sampled_ids) < sample_num:
            jsonl_id = random.randint(1, len(jsonl_files)) - 1
            line_count = wc_jsonl_lines(jsonl_files[jsonl_id])
            line_id = random.randint(1, line_count) - 1
            if (jsonl_id, line_id) not in sampled_ids:
                sampled_ids.append((jsonl_id, line_id))
    elif len(select_ids) >= sample_num:
        sampled_ids = select_ids[:sample_num]
    else:
        assert len(select_ids) >= sample_num

    # print(sampled_ids)

    select_docs = []
    for jsonl_id, line_id in sampled_ids:
        chosen_line_ids = [line_id+i for i in range(chosen_num)]
        with read_jsonl_file(jsonl_files[jsonl_id]) as js_reader:
            for di, doc in enumerate(js_reader):
                if di in chosen_line_ids:
                    select_docs.append(doc)
                
                if di > chosen_line_ids[-1]:
                    break
    
    return select_docs

def distribute_create_pre_multihop_datasets(num_process: int, recovered_files: List[str], sampled_files: List[str], context_upbound: int, skip_step: int, lang: str, dataset_dir: str, onetime_chosen_num: int):
    def generate_args_list(process_num, recovered_files, sampled_files, context_upbound, skip_step, lang, dataset_dir, onetime_chosen_num):
        process_size = math.ceil(len(recovered_files) / process_num)

        args_list = []
        for i in range(0, len(recovered_files), process_size):
            args_list.append((recovered_files[i:i+process_size], sampled_files, context_upbound, skip_step, lang, dataset_dir, onetime_chosen_num))

        return args_list
    
    args_list = generate_args_list(num_process, recovered_files, sampled_files, context_upbound, skip_step, lang, dataset_dir, onetime_chosen_num)

    with Pool(processes=len(args_list)) as pool:
        for _ in tqdm(pool.imap(create_pre_multihop_datasets, args_list), total=len(args_list)):
            pass

def create_pre_multihop_datasets(args):
    recovered_files, sampled_files, context_upbound, skip_step, lang, dataset_dir, onetime_chosen_num = args
    def token_count(doc_content, lang):
        if lang == 'zh':
            token_per_byte = 0.248
        else:
            assert lang == 'en'
            token_per_byte = 0.263
        content_bytes = len(doc_content.encode())
        return math.ceil(content_bytes * token_per_byte)

    def sampling_docs_with_longcontext_size(long_context_size: int, chunk_size: int, included_docs:List[dict], sampled_files: List[str], lang: str, drop_last: bool=False, onetime_chosen_num: int=10):
        prefix_sum_token_count = [0]
        sampled_doc_tokens = 0
        for path_doc in included_docs:
            sampled_doc_tokens += token_count(path_doc['content'], lang)
            # prefix_sum_token_count.append(sampled_doc_tokens)

        # sampled_docs = included_docs
        sampled_docs = []
        excluded_ids = [d['id'] for d in included_docs]

        sampling_exit = False
        while not sampling_exit:
            cur_sampled_docs = sample_doc(sampled_files, 1, chosen_num=onetime_chosen_num)
            cur_sampled_docs = [sdoc for sdoc in cur_sampled_docs if sdoc['id'] not in excluded_ids]
            for sdoc in cur_sampled_docs:
                sampled_doc_tokens += token_count(sdoc['content'], lang)
                prefix_sum_token_count.append(sampled_doc_tokens)
                sampled_docs.append(sdoc)
                excluded_ids.append(sdoc['id'])
                if sampled_doc_tokens >= long_context_size:
                    sampling_exit = True
                    break
                
        if drop_last:
            sampled_docs = sampled_docs[:-1]
            prefix_sum_token_count = prefix_sum_token_count[:-1]

        random.shuffle(sampled_docs)

        chunking_sampled_docs = []
        per_chunking_docs = []
        left = 0
        for right in range(1, len(prefix_sum_token_count)):
            scd = sampled_docs[right-1]
            per_chunking_docs.append({'id': scd['id'], 'content': scd['content']})

            if prefix_sum_token_count[right] - prefix_sum_token_count[left] > chunk_size:
                # chunking_sampled_docs.append(per_chunking_docs)
                chunking_sampled_docs.extend(per_chunking_docs)
                per_chunking_docs = []
                left = right
                
        if len(per_chunking_docs) > 0:
            # chunking_sampled_docs.append(per_chunking_docs)
            chunking_sampled_docs.extend(per_chunking_docs)
                
        return chunking_sampled_docs
    
    for recovered_file in recovered_files:
        with read_jsonl_file(recovered_file) as doc_reader:
            jsonl_prefix = recovered_file.split('/')[-1].split('.')[0]
            with open(os.path.join(dataset_dir, f'{jsonl_prefix}_{context_upbound}.jsonl'), 'w') as fw:
                for doc in tqdm(doc_reader):
                    if doc['type'] == 'intra_doc':
                        sampled_context_size = context_upbound - skip_step
                    else:
                        sampled_context_size = context_upbound - 2*skip_step

                    sampled_context_docs = sampling_docs_with_longcontext_size(sampled_context_size, skip_step, doc['other_docs'], sampled_files, lang, drop_last=False, onetime_chosen_num=onetime_chosen_num)
                    doc['sampled_context_docs'] = sampled_context_docs

                    fw.write(json.dumps(doc, ensure_ascii=False)+'\n')
                    fw.flush()


def format_concat_docs_with_mqa(docs: List[dict], mqa: Union[dict, str], lang: str):
    def token_count(concat_content, token_per_byte):
        content_bytes = len(concat_content.encode())
        return math.ceil(content_bytes * token_per_byte)

    if lang == 'en':
        token_per_byte = 0.263
        q = 'question'
        a = 'answer'
    else:
        token_per_byte = 0.248
        q = '问题'
        a = '回答'
        
    if lang == 'en':
        content_key = 'Passage {pi}:\n'
        # with CoT
        qa_format = 'Answer the question based on the given passages.\n\nThe following are given passages.\n{concat_content}\n\nAnswer the question based on the given passages and provide a complete reasoning process.\nQuestion:{q}\nAnswer:'
    else:
        content_key = '文章 {pi}：\n'
        # with CoT
        qa_format = '根据给定的段落回答问题。\n\n以下是给定的段落。\n{concat_content}\n\n请结合上面材料回答以下问题，并且给出完整的推理过程。\n问题：{q}\n答案：'

    concat_content = '\n'.join([content_key.format(pi=di+1)+doc['content'] for di, doc in enumerate(docs)])

    if type(mqa) is str:
        try:
            mqa = json.loads(mqa)
        except:
            print(mqa)
            return None
    
    try:
        question, answer = mqa[q], mqa[a]
    except:
        if q == 'question':
            q = '问题'
            a = '回答'
        else:
            q = 'question'
            a = 'answer'
        if q not in mqa or a not in mqa:
            return None
        
        question, answer = mqa[q], mqa[a]
        
    text = qa_format.format(concat_content=concat_content, q=question)

    return {
        'prompt': text,
        'output': answer,
        'token_count': token_count(text+answer, token_per_byte)
    }


def format_concat_docs_with_sqa(docs: List[dict], sqa: dict, lang: str):
    def token_count(concat_content, token_per_byte):
        content_bytes = len(concat_content.encode())
        return math.ceil(content_bytes * token_per_byte)
    
    if lang == 'en':
        token_per_byte = 0.263
    else:
        token_per_byte = 0.248

    q = 'question'
    a = 'answer'
        
    if lang == 'en':
        content_key = 'Passage {pi}:\n'
        qa_format = 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{concat_content}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\nQuestion:{q}\nAnswer:'
    else:
        content_key = '文章 {pi}：\n'
        qa_format = '根据给定的段落回答问题。只给我答案，不要输出任何其他单词。\n\n以下是给定的段落。\n{concat_content}\n\n请结合上面材料回答以下问题。只给我答案，不要输出任何其他单词。\n问题：{q}\n答案：'

    concat_content = '\n'.join([content_key.format(pi=di+1)+doc['content'] for di, doc in enumerate(docs)])
        
    question, answer = sqa[q], sqa[a]
        
    text = qa_format.format(concat_content=concat_content, q=question)

    return {
        'prompt': text,
        'output': answer,
        'token_count': token_count(text+answer, token_per_byte)
    }


def create_post_multihop_datasets(pre_multihop_files: List[str], post_dataset_dir: str, lang: str):
    if not os.path.exists(post_dataset_dir):
        os.makedirs(post_dataset_dir)

    inter_doc_dir = os.path.join(post_dataset_dir, 'inter_doc')
    intra_doc_dir = os.path.join(post_dataset_dir, 'intra_doc')

    os.makedirs(inter_doc_dir, exist_ok=True)
    os.makedirs(intra_doc_dir, exist_ok=True)

    for file in pre_multihop_files:
        jsonl_prefix = file.split('/')[-1].split('.')[0]
        with read_jsonl_file(file) as doc_reader:
            for doc in tqdm(doc_reader):
                if type(doc['mqa']) is str:   
                    doc['mqa'] = re.findall(r'({.*?})', doc['mqa'])

                    if len(doc['mqa']) == 0:
                        continue
                    try:
                        doc['mqa'] = json.loads(doc['mqa'][0])
                    except:
                        continue

                if doc['type'] == 'intra_doc':
                    d1 = doc['doc1']
                    
                    chunk_size = len(doc['sampled_context_docs'])
                    insert_position = chunk_size + 1
                    for ip in range(insert_position):
                        concat_docs = copy.deepcopy(doc['sampled_context_docs'])

                        if ip == 0:
                            concat_docs[0] = [d1] + concat_docs[0]
                        else:
                            concat_docs[ip-1] = concat_docs[ip-1] + [d1]

                        flatten_concat_docs = []
                        for cds in concat_docs:
                            flatten_concat_docs.extend(cds)

                        instruct_format = format_concat_docs_with_mqa(flatten_concat_docs, doc['mqa'], lang)
                        if instruct_format is None:
                            continue

                        dataset_item = {
                            'concat_docs': flatten_concat_docs,
                            'clue_docs': [d1],
                            'mqa': doc['mqa'],
                            'prompt': instruct_format['prompt'],
                            'output': instruct_format['output'],
                            'token_count': instruct_format['token_count']
                        }
                        intra_local_doc_dir = os.path.join(intra_doc_dir, f'p{ip}')
                        os.makedirs(intra_local_doc_dir, exist_ok=True)
                        
                        with open(os.path.join(intra_local_doc_dir, f'{jsonl_prefix}_p{ip}.jsonl'), 'a') as fw:
                            fw.write(json.dumps(dataset_item, ensure_ascii=False)+'\n')
                else:
                    d1 = doc['doc1']
                    d2 = doc['doc2']
                    
                    chunk_size = len(doc['sampled_context_docs'])
                    insert_position = chunk_size + 1
                    for ip1 in range(insert_position):
                        concat_docs = copy.deepcopy(doc['sampled_context_docs'])
                        if ip1 == 0:
                            concat_docs[0] = [d1] + concat_docs[0]
                        else:
                            concat_docs[ip1-1] = concat_docs[ip1-1] + [d1]

                        for ip2 in range(ip1, insert_position):
                            ip1_concat_docs = copy.deepcopy(concat_docs)
                            if ip2 == 0:
                                ip1_concat_docs[0] = [d2] + ip1_concat_docs[0]
                            else:
                                ip1_concat_docs[ip2-1] = ip1_concat_docs[ip2-1] + [d2]

                            flatten_concat_docs = []
                            for cds in ip1_concat_docs:
                                flatten_concat_docs.extend(cds)

                            instruct_format = format_concat_docs_with_mqa(flatten_concat_docs, doc['mqa'], lang)
                            if instruct_format is None:
                                continue
                            dataset_item = {
                                'concat_docs': flatten_concat_docs,
                                'clue_docs': [d1, d2],
                                'mqa': doc['mqa'],
                                'prompt': instruct_format['prompt'],
                                'output': instruct_format['output'],
                                'token_count': instruct_format['token_count']
                            }
                            inter_local_doc_dir = os.path.join(inter_doc_dir, f'p{ip1}_p{ip2}')
                            os.makedirs(inter_local_doc_dir, exist_ok=True)

                            with open(os.path.join(inter_local_doc_dir, f'{jsonl_prefix}_p{ip1}_p{ip2}.jsonl'), 'a') as fw:
                                fw.write(json.dumps(dataset_item, ensure_ascii=False)+'\n')


def create_multihop_train_datasets(pre_multihop_files: List[str], train_dataset_dir: str, lang: str, token_limit_lowerbound: int=4096, token_limit_upperbound: int=32768):
    def token_count(concat_content, token_per_byte):
        content_bytes = len(concat_content.encode())
        return math.ceil(content_bytes * token_per_byte)
    
    if lang == 'en':
        token_per_byte = 0.263
    else:
        token_per_byte = 0.248

    def insert_docs(doc_list, insert_docs, token_limit):
        prefixsum_doc_tokens = [0]
        for ad in doc_list:
            prefixsum_doc_tokens.append(prefixsum_doc_tokens[-1]+token_count(ad['content'], token_per_byte))
            
        insert_doc_len = sum([token_count(d['content'], token_per_byte) for d in insert_docs])
        for adi, ad_len in enumerate(prefixsum_doc_tokens):
            if ad_len + insert_doc_len > token_limit:
                break

        doc_list = doc_list[:adi]
        ips = []
        for idoc in insert_docs:
            ip = random.randint(0, len(doc_list))
            ips.append(ip)
            doc_list = doc_list[:ip] + [idoc] + doc_list[ip:]
        return doc_list, ips

    os.makedirs(train_dataset_dir, exist_ok=True)
    inter_doc_dir = os.path.join(train_dataset_dir, f'inter_doc_{lang}')
    intra_doc_dir = os.path.join(train_dataset_dir, f'intra_doc_{lang}')
    os.makedirs(inter_doc_dir, exist_ok=True)
    os.makedirs(intra_doc_dir, exist_ok=True)

    statistic_span = token_limit_upperbound // 2048 + 1
    statistics = {
        'inter_less_token_limit': 0,
        'inter_over_token_limit': 0,
        'inter_token_distribution': [0] * statistic_span,
        'intra_less_token_limit': 0,
        'intra_over_token_limit': 0,
        'intra_token_distribution': [0] * statistic_span
    }
    
    for file in pre_multihop_files:
        print(file)
        domain_name = file.split('/')[-2]

        try:
            with read_jsonl_file(file) as doc_reader:
                for doc in tqdm(doc_reader):
                    if type(doc['mqa']) is str:
                        doc['mqa'] = re.findall(r'({.*?})', doc['mqa'])
                        if len(doc['mqa']) == 0:
                            continue
                        try:
                            doc['mqa'] = json.loads(doc['mqa'][0])
                        except:
                            continue

                    aggregate_docs = copy.deepcopy(doc['other_docs'])
                    close_doc_tokens = sum([token_count(d['content'], token_per_byte) for d in aggregate_docs])
                    aggregate_docs = aggregate_docs + doc.get('sampled_context_docs', [])

                    token_limit = random.randint(token_limit_lowerbound, token_limit_upperbound)
                    token_limit = max(token_limit, close_doc_tokens)
                    
                    if doc['type'] == 'intra_doc':
                        # d1 = doc['doc1']
                        flatten_concat_docs, ips = insert_docs(aggregate_docs, [doc['clue_docs'][0]], token_limit)

                        instruct_format = format_concat_docs_with_mqa(flatten_concat_docs, doc['mqa'], lang)
                        if instruct_format is None:
                            continue
                        
                        dataset_item = {
                            'hop': doc['hop'],
                            'concat_docs': flatten_concat_docs,
                            'clue_docs': [doc['clue_docs'][0]],
                            'mqa': doc['mqa'],
                            'prompt': instruct_format['prompt'],
                            'output': instruct_format['output'],
                            'token_count': instruct_format['token_count'],
                            'insert_position': ips,
                            'domain': domain_name
                        }

                        if instruct_format['token_count'] > token_limit_upperbound:
                            statistics['intra_over_token_limit'] += 1
                        else:
                            statistics['intra_less_token_limit'] += 1

                        tci = math.ceil(instruct_format['token_count'] / 2048)
                        tci = min(statistic_span, tci)
                        statistics['intra_token_distribution'][tci-1] += 1
                        
                        with open(os.path.join(intra_doc_dir, f'{domain_name}_train.jsonl'), 'a') as fw:
                            fw.write(json.dumps(dataset_item, ensure_ascii=False)+'\n')
                    else:
                        # d1 = doc['doc1']
                        # d2 = doc['doc2']
                        flatten_concat_docs, ips = insert_docs(aggregate_docs, doc['clue_docs'], token_limit)

                        instruct_format = format_concat_docs_with_mqa(flatten_concat_docs, doc['mqa'], lang)
                        if instruct_format is None:
                            continue

                        dataset_item = {
                            'hop': doc['hop'],
                            'concat_docs': flatten_concat_docs,
                            'clue_docs': doc['clue_docs'],
                            'mqa': doc['mqa'],
                            'prompt': instruct_format['prompt'],
                            'output': instruct_format['output'],
                            'token_count': instruct_format['token_count'],
                            'insert_position': ips,
                            'domain': domain_name
                        }

                        if instruct_format['token_count'] > token_limit_upperbound:
                            statistics['inter_over_token_limit'] += 1
                        else:
                            statistics['inter_less_token_limit'] += 1

                        tci = math.ceil(instruct_format['token_count'] / 2048)
                        tci = min(statistic_span, tci)
                        statistics['inter_token_distribution'][tci-1] += 1

                        with open(os.path.join(inter_doc_dir, f'{domain_name}_train.jsonl'), 'a') as fw:
                            fw.write(json.dumps(dataset_item, ensure_ascii=False)+'\n')
        except:
            continue

    print(json.dumps(statistics, indent=4))


def create_singlehop_train_datasets(pre_multihop_files: List[str], train_dataset_dir: str, lang: str, token_limit_lowerbound: int=4096, token_limit_upperbound: int=32768):
    def token_count(concat_content, token_per_byte):
        content_bytes = len(concat_content.encode())
        return math.ceil(content_bytes * token_per_byte)
    
    if lang == 'en':
        token_per_byte = 0.263
    else:
        token_per_byte = 0.248

    def insert_docs(doc_list, insert_docs, token_limit):
        prefixsum_doc_tokens = [0]
        for ad in doc_list:
            prefixsum_doc_tokens.append(prefixsum_doc_tokens[-1]+token_count(ad['content'], token_per_byte))
            
        insert_doc_len = sum([token_count(d['content'], token_per_byte) for d in insert_docs])
        for adi, ad_len in enumerate(prefixsum_doc_tokens):
            if ad_len + insert_doc_len > token_limit:
                break

        doc_list = doc_list[:adi]
        ips = []
        for idoc in insert_docs:
            ip = random.randint(0, len(doc_list))
            ips.append(ip)
            doc_list = doc_list[:ip] + [idoc] + doc_list[ip:]
        return doc_list, ips

    os.makedirs(train_dataset_dir, exist_ok=True)
    single_doc_dir = os.path.join(train_dataset_dir, f'single_doc_{lang}')
    os.makedirs(single_doc_dir, exist_ok=True)

    statistic_span = token_limit_upperbound // 2048 + 1
    statistics = {
        'inter_less_token_limit': 0,
        'inter_over_token_limit': 0,
        'inter_token_distribution': [0] * statistic_span,
        'intra_less_token_limit': 0,
        'intra_over_token_limit': 0,
        'intra_token_distribution': [0] * statistic_span
    }
    for file in pre_multihop_files:
        domain_name = file.split('/')[-2]

        try:
            with read_jsonl_file(file) as doc_reader:
                for doc in tqdm(doc_reader):
                    aggregate_docs = copy.deepcopy(doc['other_docs'])
                    close_doc_tokens = sum([token_count(d['content'], token_per_byte) for d in aggregate_docs])
                    aggregate_docs = aggregate_docs + doc.get('sampled_context_docs', [])

                    token_limit = random.randint(token_limit_lowerbound, token_limit_upperbound)
                    token_limit = max(token_limit, close_doc_tokens)
                    
                    if doc['type'] == 'intra_doc':
                        d1 = doc['clue_docs'][0]

                        flatten_concat_docs, ips = insert_docs(aggregate_docs, [d1], token_limit)
                        single_qa = doc['clue_qas'][0]

                        instruct_format = format_concat_docs_with_sqa(flatten_concat_docs, single_qa, lang)
                        if instruct_format is None:
                            continue
                        
                        dataset_item = {
                            'concat_docs': flatten_concat_docs,
                            'clue_docs': [d1],
                            'sqa': single_qa,
                            'prompt': instruct_format['prompt'],
                            'output': instruct_format['output'],
                            'token_count': instruct_format['token_count'],
                            'insert_position': ips,
                            'domain': domain_name
                        }

                        if instruct_format['token_count'] > token_limit_upperbound:
                            statistics['intra_over_token_limit'] += 1
                        else:
                            statistics['intra_less_token_limit'] += 1

                        tci = math.ceil(instruct_format['token_count'] / 2048)
                        tci = min(statistic_span, tci)
                        statistics['intra_token_distribution'][tci-1] += 1
                        
                        with open(os.path.join(single_doc_dir, f'{domain_name}_train.jsonl'), 'a') as fw:
                            fw.write(json.dumps(dataset_item, ensure_ascii=False)+'\n')
                    else:
                        single_qas = doc['clue_qas']
                        clue_docs = doc['clue_docs']

                        for qai, di in zip(single_qas, clue_docs):
                            flatten_concat_docs, ips = insert_docs(aggregate_docs, [di], token_limit)
                            instruct_format = format_concat_docs_with_sqa(flatten_concat_docs, qai, lang)
                            if instruct_format is None:
                                continue

                            dataset_item = {
                                'concat_docs': flatten_concat_docs,
                                'clue_docs': [di],
                                'sqa': qai,
                                'prompt': instruct_format['prompt'],
                                'output': instruct_format['output'],
                                'token_count': instruct_format['token_count'],
                                'insert_position': ips,
                                'domain': domain_name
                            }

                            if instruct_format['token_count'] > token_limit_upperbound:
                                statistics['inter_over_token_limit'] += 1
                            else:
                                statistics['inter_less_token_limit'] += 1

                            tci = math.ceil(instruct_format['token_count'] / 2048)
                            tci = min(statistic_span, tci)
                            statistics['inter_token_distribution'][tci-1] += 1

                            with open(os.path.join(single_doc_dir, f'{domain_name}_train.jsonl'), 'a') as fw:
                                fw.write(json.dumps(dataset_item, ensure_ascii=False)+'\n')
        except:
            continue
    print(json.dumps(statistics, indent=4))


def sampling_perturbation_train_datasets(target_dir, save_dir, context_upbound, sample_upbound=10000):
    jsonl_files = glob(f'{target_dir}/*/*/*')
    for jsonl_file in tqdm(jsonl_files):
        lang, task_mode, file_name = jsonl_file.split('/')[-3],jsonl_file.split('/')[-2], jsonl_file.split('/')[-1]
        
        save_task_dir = os.path.join(save_dir, lang)
        os.makedirs(save_task_dir, exist_ok=True)
        save_task_dir = os.path.join(save_task_dir, task_mode)
        os.makedirs(save_task_dir, exist_ok=True)

        save_file = os.path.join(save_task_dir, file_name)
        sample_cnt = 0
        with open(save_file, 'w') as fw:
            with read_jsonl_file(jsonl_file) as doc_reader:
                for doc in doc_reader:
                    if doc['token_count'] > context_upbound:
                        continue

                    sample_cnt += 1
                    fw.write(json.dumps(doc, ensure_ascii=False)+'\n')

                    if sample_cnt > sample_upbound:
                        break


if __name__ == '__main__':
    from tqdm import tqdm

    # # recover multihop QA information as single samples
    # jsonl_dir = '/cpfs01/shared/public/chenzhi/datasets/longmit/zh/zh-ebook-longmit-4hop/*'
    # recover_save_dir = '/cpfs01/shared/public/chenzhi/datasets/longmit/zh/zh-ebook-longmit-4hop-recover'
    # os.makedirs(recover_save_dir, exist_ok=True)

    # jsonl_files = glob(jsonl_dir)
    # for file in tqdm(jsonl_files):
    #     recover_multihop_info(file, recover_save_dir)

    # pre_dataset_dir = '/cpfs01/shared/public/chenzhi/datasets/longmit-pre/en-stackexchange'
    # os.makedirs(pre_dataset_dir, exist_ok=True)

    # recovered_files = glob(f'{recover_save_dir}/*')
    # sampled_files = glob('/cpfs01/shared/alillm2/user/zhangshuo/datasets/official_Ampere2.5_7B_2.3.0/cn/zh-ebook/*')
    context_upbound = 131072
    # skip_step = 4096
    # lang = 'en'
    # process_num = 10
    # onetime_chosen_num = 2
    # distribute_create_pre_multihop_datasets(process_num, recovered_files, sampled_files, context_upbound, skip_step, lang, pre_dataset_dir, onetime_chosen_num)

    # pre_multihop_files = glob(f'{pre_dataset_dir}/*')

    # token_limit_lowerbound = 4096
    # token_limit_upperbound = 131072
    # train_dataset_dir = f'/cpfs01/shared/public/chenzhi/datasets/longmit-train/{lang}'
    # os.makedirs(train_dataset_dir, exist_ok=True)
    # create_multihop_train_datasets(pre_multihop_files, train_dataset_dir, lang, token_limit_lowerbound, token_limit_upperbound)
    # create_singlehop_train_datasets(pre_multihop_files, train_dataset_dir, lang, token_limit_lowerbound, token_limit_upperbound)

    train_dataset_dir = f'/cpfs01/shared/public/chenzhi/datasets/longmit-train'
    sample_upbound = 5000
    perturbation_train_dir = '/cpfs01/shared/public/chenzhi/datasets/longmit-perturbation-train'
    sampling_perturbation_train_datasets(train_dataset_dir, perturbation_train_dir, context_upbound, sample_upbound)