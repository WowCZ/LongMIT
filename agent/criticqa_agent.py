'''
Description: 
Version: 1.0
Autor: Zhi Chen
Date: 2024-07-17 12:16:34
LastEditors: chenzhi chenzhi@pjlab.org.cn
LastEditTime: 2024-08-01 08:46:36
'''

import os, sys
sys.path.append(os.getcwd())

import json
import time
import copy
import importlib
import subprocess
from glob import glob
from agent.utils import *
from agent.base_agent import AgentFlow
from iclm.data_process import read_jsonl_file


class CriticQAAgent(AgentFlow):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.jsonl_files = glob(self.cfg.datasets)
        self.api_cfg = self.cfg.llm_api
        self.save_dir = self.cfg.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.prompts = importlib.import_module(f'agent.prompts.{self.cfg.domain}')

    def run(self):        
        for jsonl_file in self.jsonl_files:
            domain_name, fname = jsonl_file.split('/')[-2:]
            save_dir = os.path.join(self.save_dir, domain_name)
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, fname)

            try:
                recorded_cnt = subprocess.run(['wc', '-l', save_file], capture_output=True, text=True)
                has_recorded = int(recorded_cnt.stdout.split()[0])
            except:
                has_recorded = 0
            has_recorded -= 1

            with open(save_file, 'a') as fw:
                end_id = 0
                with read_jsonl_file(jsonl_file) as doc_reader:
                    for doc_with_qa in doc_reader:
                        if has_recorded >= end_id:
                            end_id += 1
                            continue

                        stime = time.time()
                        format_a = self._rewrite_answer(doc_with_qa)
                        agg_time = time.time()
                        print(f'>>> format_answer spent {agg_time-stime}s')
                        format_doc_with_qa = copy.deepcopy(doc_with_qa)
                        format_doc_with_qa['output'] = format_a
                        format_doc_with_qa['dedup_output'] = doc_with_qa['output']

                        fw.write(json.dumps(format_doc_with_qa, ensure_ascii=False)+'\n')
                        fw.flush()
    
    def _rewrite_answer(self, doc_with_qa):
        prompt = doc_with_qa['prompt']
        output = doc_with_qa['output']

        question = prompt.split('\n')[-2]
        assert question.startswith('问题') or question.startswith('Question')

        qa_pair = {'question': question, 'answer': output}
        format_answer = rewrite_answer(qa_pair, self.cfg.lang, self.api_cfg.model_name, self.api_cfg.api_base, self.prompts.rewrite_answer_prompt)

        return format_answer
    
    def _score_qa_pair(self, qa_pair):
        pass
    

if __name__ == '__main__':
    agent_config = 'agent/configs/criticqa_en_ra.yaml'
    longqa_generator = CriticQAAgent(agent_config)
    longqa_generator.run()

    # /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding/bin/python agent/criticqa_agent.py



