'''
Description: 
Version: 1.0
Autor: Zhi Chen
Date: 2024-07-17 12:16:34
LastEditors: chenzhi chenzhi@pjlab.org.cn
LastEditTime: 2024-08-07 03:43:50
'''

import os, sys
sys.path.append(os.getcwd())

import json
import time
import math
import random
import importlib
import subprocess
from glob import glob
from typing import List
from agent.utils import *
from agent.base_agent import AgentFlow
from doc_process.data_process import read_jsonl_file
from doc_process.embed_docs import initial_embedder


class LongQAAgent(AgentFlow):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.jsonl_files = glob(self.cfg.datasets)
        self.aggregate_cfg = self.cfg.pipelines.aggregate_docs
        self.extractq_cfg = self.cfg.pipelines.extract_questions
        self.generatea_cfg = self.cfg.pipelines.generate_answers
        self.mqa_cfg = self.cfg.pipelines.aggregate_multihop_qas
        self.api_cfg = self.cfg.llm_api
        self.embedder_cfg = self.cfg.pipelines.calculate_qas_correlation.embedder
        self.save_dir = self.cfg.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
            
        self.prompts = importlib.import_module(f'agent.prompts.{self.cfg.domain}')

    def run(self):
        self.embedder = initial_embedder(self.embedder_cfg)
        
        for jsonl_file in self.jsonl_files:
            fname = jsonl_file.split('/')[-1]
            save_file = os.path.join(self.save_dir, fname)
            try:
                recorded_cnt = subprocess.run(['wc', '-l', save_file], capture_output=True, text=True)
                has_recorded = int(recorded_cnt.stdout.split()[0])
            except:
                has_recorded = 0
            has_recorded -= 1

            with open(save_file, 'a') as fw:
                last_docs = False
                end_id = 0
                while not last_docs:
                    if has_recorded >= end_id:
                        end_id += 1
                        continue

                    stime = time.time()
                    aggregated_docs, end_id, aggregated_token_cnt, last_docs = self._aggregate_docs(jsonl_file, end_id)
                    agg_time = time.time()
                    print(f'>>> aggregate_docs spent {agg_time-stime}s')
                    all_extracted_questions, all_chunks = self._extract_questions(aggregated_docs)
                    exa_time = time.time()
                    qcnt = 0
                    for chunk_qs in all_extracted_questions:
                        for qs in chunk_qs:
                            qcnt += len(qs)
                    print(f'>>> extract_questions spent {exa_time-agg_time}s for {qcnt} questions')
                    reserved_questions = self._filter_questions(all_extracted_questions)
                    fil_time = time.time()
                    qcnt = 0
                    for chunk_qs in reserved_questions:
                        for qs in chunk_qs:
                            qcnt += len(qs)
                    print(f'>>> filter_questions spent {fil_time-exa_time}s for reserved {qcnt} questions')
                    qa_pairs = self._generate_answers(all_chunks, reserved_questions)
                    ga_time = time.time()
                    qacnt = 0
                    for chunk_qs in qa_pairs:
                        for qs in chunk_qs:
                            qacnt += len(qs)
                    print(f'>>> generate_answers spent {ga_time-fil_time}s for {qacnt} qa pairs')
                    if qacnt == 0:
                        end_id += 1
                        continue
                    clustered_qa_pair_info, qas_per_doc = self._calculate_qas_correlation(qa_pairs)
                    clu_time = time.time()
                    print(f'>>> calculate_qas_correlation spent {clu_time-ga_time}s')
                    multihop_qa_pairs, selected_qa_ids = self._aggregate_multihop_qas(qas_per_doc, clustered_qa_pair_info)
                    mqa_time = time.time()
                    print(f'>>> aggregate_multihop_qas spent {mqa_time-clu_time}s')
                    print(multihop_qa_pairs)

                    record_mqa_info = dict(
                        aggregated_docs=aggregated_docs,
                        intra_doc_qas=qa_pairs,
                        multihop_qas=multihop_qa_pairs,
                        selected_qa_ids=selected_qa_ids,
                        clustered_mqa_info=clustered_qa_pair_info,
                        token_count=aggregated_token_cnt
                    )

                    fw.write(json.dumps(record_mqa_info, ensure_ascii=False)+'\n')
                    fw.flush()

    def _aggregate_docs(self, jsonl_file: str, start_id: int):
        '''
        description: 
        return {*}
        '''        
        assert self.aggregate_cfg.type in ['random', 'iclm']

        def token_count(content, token_per_byte):
            content_bytes = len(content.encode())
            return math.ceil(content_bytes * token_per_byte)
        
        if self.cfg.lang == 'en':
            token_per_byte = 0.263
        else:
            token_per_byte = 0.248

        limited_doc_tokens = random.randint(self.aggregate_cfg.min_aggregated_tokens, self.aggregate_cfg.max_aggregated_tokens)
        if self.aggregate_cfg.type == 'random':
            aggregated_docs = []
            aggregated_token_cnt = 0
            end_id = start_id
            last_docs = False
            with read_jsonl_file(jsonl_file) as doc_reader:
                for di, doc in enumerate(doc_reader):
                    if di < start_id:
                        continue
                    
                    end_id += 1
                    aggregated_token_cnt += token_count(doc['content'], token_per_byte)
                    aggregated_docs.append(doc)

                    if aggregated_token_cnt >= limited_doc_tokens:
                        return aggregated_docs, end_id, aggregated_token_cnt, last_docs
        else:
            self.aggregate_cfg.type == 'iclm'
            aggregated_docs = []
            aggregated_token_cnt = 0
            end_id = start_id
            last_docs = False
            with read_jsonl_file(jsonl_file) as doc_reader:
                for di, doc_path in enumerate(doc_reader):
                    if di < start_id:
                        continue

                    end_id += 1
                    for did, dcontent in zip(doc_path['id_path'], doc_path['content_path']):
                        aggregated_token_cnt += token_count(dcontent, token_per_byte)
                        aggregated_docs.append({'id': did, 'content': dcontent})

                    return aggregated_docs, end_id, aggregated_token_cnt, last_docs
                
        last_docs = True
        return aggregated_docs, end_id, aggregated_token_cnt, last_docs
    
    def _extract_questions(self, docs: List[dict]):
        all_extracted_questions = []
        all_chunks = []
        for doc in docs:
            questions, chunks = extract_questions(doc['content'], self.cfg.lang, self.api_cfg.model_name, self.api_cfg.api_base, self.extractq_cfg.chunk_size, self.prompts.extract_question_prompt)
            all_extracted_questions.append(questions)
            all_chunks.append(chunks)

        return all_extracted_questions, all_chunks
    
    def _filter_questions(self, extracted_questions):
        filtered_labels = filter_questions(extracted_questions, self.cfg.lang, self.api_cfg.model_name, self.api_cfg.api_base, self.prompts.filter_question_prompt)
        reserved_questions = []
        for doc_qs, doc_flabels in zip(extracted_questions, filtered_labels):
            reserved_doc_questions = []
            for chunk_qs, chunk_flabels in zip(doc_qs, doc_flabels):
                reserved_chunk_questions = []
                for q, fl in zip(chunk_qs, chunk_flabels):
                    if fl:
                        continue
                    reserved_chunk_questions.append(q)
                reserved_doc_questions.append(reserved_chunk_questions)
            reserved_questions.append(reserved_doc_questions)
        
        return reserved_questions
    
    def _generate_answers(self, doc_chunks, questions):
        all_qa_pairs = []
        for chunks, chunk_questions in zip(doc_chunks, questions):
            doc_qa_pairs = []
            assert len(chunks) == len(chunk_questions)
            for chunk, qs in zip(chunks, chunk_questions):
                chunk_qa_pairs = []
                for q in qs:
                    a = generate_anwer(q, chunk, self.cfg.lang, self.api_cfg.model_name, self.api_cfg.api_base, self.prompts.generate_answer_prompt)
                    if self.generatea_cfg.score_qa_pairs:
                        raise NotImplementedError
                    
                    chunk_qa_pairs.append({
                        'question': q,
                        'answer': a
                    })

                if self.generatea_cfg.simplify_answer:
                    simplified_qa_pairs = simplify_qa(chunk_qa_pairs, self.cfg.lang, self.api_cfg.model_name, self.api_cfg.api_base, self.prompts.simplify_qa_prompt)
                    for qi, sqa in enumerate(simplified_qa_pairs):
                        chunk_qa_pairs[qi]['rationale'] = chunk_qa_pairs[qi]['answer']
                        chunk_qa_pairs[qi]['answer'] = sqa['answer']

                doc_qa_pairs.append(chunk_qa_pairs)
            all_qa_pairs.append(doc_qa_pairs)
        return all_qa_pairs
    
    def _calculate_qas_correlation(self, qa_pairs):
        qas_per_doc = []
        for doc_qa_pairs in qa_pairs:
            all_doc_qas = []
            for chunk_qa_pairs in doc_qa_pairs:
                all_doc_qas.extend(chunk_qa_pairs)
            qas_per_doc.append(all_doc_qas)

        similarity_matrix = calc_similarity_matrix(qas_per_doc, self.embedder, self.embedder_cfg, self.cfg.lang)
        clustered_qa_pair_info = cluster_qa_pairs(qas_per_doc, similarity_matrix, self.mqa_cfg.topk)
        return clustered_qa_pair_info, qas_per_doc
    
    def _aggregate_multihop_qas(self, qa_pairs, clustered_qa_pair_info):
        def recover_mqa(mqa: str):
            mqa = re.findall(r'({.*?})', mqa)
            if len(mqa) == 0:
                return {}
            try:
                mqa = json.loads(mqa[0])
            except:
                mqa = {}
            return mqa

        multihop_qa_pairs = []
        qas = []
        for qa in qa_pairs:
            qas.extend(qa)

        selected_len = min(len(qas), 10)
        selected_qa_ids = random.sample(list(range(len(qas))), selected_len)
        for qi, qa1 in enumerate(qas):
            if qi not in selected_qa_ids:
                continue

            in_doc_qas = []
            for knn_id in clustered_qa_pair_info[qi]['indoc_ids']:
                in_doc_qas.append(qas[knn_id])
            
            cross_doc_qas = []
            for knn_id in clustered_qa_pair_info[qi]['crossdoc_ids']:
                cross_doc_qas.append(qas[knn_id])

            indoc_multihop_qa_pairs = []
            tmp_qa = qa1
            for qa2 in in_doc_qas:
                indoc_multihop_qa_pair = merge_qa_couples(tmp_qa, qa2, self.cfg.lang, self.api_cfg.model_name, self.api_cfg.api_base, self.prompts.merge_qa_prompt)
                indoc_multihop_qa_pair = recover_mqa(indoc_multihop_qa_pair)
                if len(indoc_multihop_qa_pair) == 0:
                    continue
                indoc_multihop_qa_pairs.append(indoc_multihop_qa_pair)
                tmp_qa = indoc_multihop_qa_pair

            crossdoc_multihop_qa_pairs = []
            tmp_qa = qa1
            for qa2 in cross_doc_qas:
                crossdoc_multihop_qa_pair = merge_qa_couples(tmp_qa, qa2, self.cfg.lang, self.api_cfg.model_name, self.api_cfg.api_base, self.prompts.merge_qa_prompt)
                # crossdoc_multihop_qa_pairs.append(crossdoc_multihop_qa_pair)
                crossdoc_multihop_qa_pair = recover_mqa(crossdoc_multihop_qa_pair)
                if len(crossdoc_multihop_qa_pair) == 0:
                    continue
                crossdoc_multihop_qa_pairs.append(crossdoc_multihop_qa_pair)
                tmp_qa = crossdoc_multihop_qa_pair

            multihop_qa_pairs.append(
                {
                    'indoc': indoc_multihop_qa_pairs,
                    'crossdoc': crossdoc_multihop_qa_pairs
                }
            )
        
        return multihop_qa_pairs, selected_qa_ids

if __name__ == '__main__':
    agent_config = 'agent/configs/longqa_kepuchina_random.yaml'
    longqa_generator = LongQAAgent(agent_config)
    longqa_generator.run()

    # /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding/bin/python agent/longqa_agent.py



