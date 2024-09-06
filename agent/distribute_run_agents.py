'''
Description: 
Version: 1.0
Autor: Zhi Chen
Date: 2024-07-17 13:33:52
LastEditors: chenzhi chenzhi@pjlab.org.cn
LastEditTime: 2024-07-31 23:15:23
'''
import os, sys
sys.path.append(os.getcwd())

from agent.longqa_agent import LongQAAgent
from agent.criticqa_agent import CriticQAAgent
from multiprocessing import Pool
from omegaconf import OmegaConf
from glob import glob
import math
from tqdm import tqdm

def run_agent(args):
    gpu_rank, jsonl_files, config_file, mode = args
    if mode == 'longqa':
        agent = LongQAAgent(config_file)
        agent.embedder_cfg.device = f'{agent.embedder_cfg.device}:{gpu_rank}'
    elif mode == 'criticqa':
        agent = CriticQAAgent(config_file)

    agent.jsonl_files = jsonl_files
    agent.run()

def distribute_agents(num_process: int, gpu_num: int, config_file: str, mode: str):

    distribut_process_gpus = [i%gpu_num for i in range(num_process)]
    agent_cfg = OmegaConf.load(config_file)
    jsonl_files = glob(agent_cfg.datasets)

    file_size_per_process = math.ceil(len(jsonl_files) / num_process)

    args_list = []
    for i, fi in enumerate(range(0, len(jsonl_files), file_size_per_process)):
        args_list.append((distribut_process_gpus[i], jsonl_files[fi: fi+file_size_per_process], config_file, mode))

    with Pool(len(args_list)) as agent_pool:
        for _ in tqdm(agent_pool.imap(run_agent, args_list), total=len(args_list)):
            pass

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='embed_docs')
    parser.add_argument('--config', type=str, default='agent/configs/longqa_example.yaml', help='longqa agent config')
    parser.add_argument('--num_process', type=int, default=16, help='Number of Threads')
    parser.add_argument('--mode', type=str, default='longqa', help='Name of agents')
    parser.add_argument('--world_rank', type=int, default=8, help='Number of GPUs')
    args = parser.parse_args()

    distribute_agents(args.num_process, args.world_rank, args.config, args.mode)


    

