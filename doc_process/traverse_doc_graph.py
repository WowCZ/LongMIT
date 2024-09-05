import os, sys
sys.path.append(os.getcwd())

from tqdm import tqdm
import argparse
from pathlib import Path
import random
from multiprocessing import Pool
import multiprocessing
import numpy as np
import random
from collections import defaultdict
import json

from faiss_knn.utils import (
    load_config,
)

from check_match import *

random.seed(0)

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# compare two documents using n-gram similarity
def generate_ngrams(text, n):
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        ngrams.add(ngram)
    return ngrams

def ngram_similarity(doc1, doc2, n):
    ngrams_doc1 = generate_ngrams(doc1, n)
    ngrams_doc2 = generate_ngrams(doc2, n)
    return jaccard_similarity(ngrams_doc1, ngrams_doc2)

def get_text_files(cfg, text_dir, domain):
    fname = []
    for f in cfg["datasets"][domain]["files"]:
        cur_text_dir = text_dir
        if text_dir.split('/')[-1] != f['domain']:
            cur_text_dir = os.path.join(text_dir, f['domain'])
            
        fname.append(os.path.join(cur_text_dir, f['name']).replace('.npy', '.jsonl'))

    return fname


class sort_class():
    def __init__(self, id2file_pos, output_file, context_len, text_key, text_files, knn_dir, faiss_cfg, merge_path):
        self.num_docs = len(id2file_pos)
        self.seen_docs = set()
        self.unseen_docs = set(range(self.num_docs))
        print(f"num docs: {self.num_docs}")
        self.id2file_pos = id2file_pos
    
        self.doc_sim_threshold = 0.75 # 0.85
        self.n = 3
        self.context_len = context_len
        self.output_file = output_file
        self.text_key = text_key
        self.text_files = text_files
        self.knn_dir = knn_dir
        self.file_name_base = "I0000000000"

        self.cur_k = None
        self.filter_docs = []
        
        self.cluster2docs = defaultdict(list)
        self.doc2cluster = {}

        self.num_docs_per_file = 50000000

        self.all_knns = []
        self.cluster_size = 21
        self.faiss_cfg = faiss_cfg
        self.merge_path = merge_path

    def load_all_knns(self):
        ivf, pq, nprob = self.faiss_cfg['ivf'], self.faiss_cfg['pq'], self.faiss_cfg['nprob']
        for i in range(0, self.num_docs, self.num_docs_per_file):
            file_name = self.file_name_base[:-len(str(i))] + str(i)
            knns = np.load(f"{self.knn_dir}/{file_name}_IVF{ivf}_PQ{pq}_np{nprob}.npy", mmap_mode="r")
            self.all_knns.append(knns)
            print(f"load knn: {i}")

    def dump_filter_docs(self):
        pickle_dump(self.filter_docs, f"{self.output_file}/filtered_docs.pkl")

    def load_corresponding_knns(self, query_id, num_docs_per_file):
        # start = time.time()
        file_id = (query_id // num_docs_per_file) 
        relative_id = query_id % num_docs_per_file
        knns = self.all_knns[file_id]
        # print(f"load knn time: {time.time() - start}")
        return knns, relative_id

    def sort(self):
        # cluster
        cluster_id = 0
        cur_cluster_len = 0

        # first doc
        self.cur_k = self.unseen_docs.pop()
        self.cluster2docs[cluster_id].append(self.cur_k)
        self.doc2cluster[self.cur_k] = cluster_id
        self.seen_docs.add(self.cur_k)
        with tqdm(total=self.num_docs-1) as pbar:
             while self.unseen_docs:
                # start_time = time.time()
                knns, relative_id = self.load_corresponding_knns(self.cur_k, self.num_docs_per_file)
                # print(f"load knn time: {time.time() - start_time}")

                # start_time = time.time()
                knn = knns[relative_id, :]
                # print(f"get knn time: {time.time() - start_time}")

                # start_time = time.time()
                first_doc = self.output_first_doc_knn(knn)
                # print(f"first doc time: {time.time() - start_time}")

                if (first_doc is None) or (cur_cluster_len >= self.cluster_size):
                    # start_time = time.time()
                    self.cur_k = self.unseen_docs.pop()
                    # print(f"random time: {time.time() - start_time}")
                    cluster_id += 1
                    cur_cluster_len = 0
                else:
                    self.cur_k = first_doc
                    self.unseen_docs.remove(self.cur_k)
                # start_time = time.time()
                self.cluster2docs[cluster_id].append(self.cur_k)
                self.doc2cluster[self.cur_k] = cluster_id
                cur_cluster_len += 1
                self.seen_docs.add(self.cur_k)
                pbar.update(1)

        pickle_dump(self.cluster2docs, f"{self.output_file}/cluster2docs.pk")
        pickle_dump(self.doc2cluster, f"{self.output_file}/doc2cluster.pk")
    
    def build_doc2_cluster(self):
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
        for cluster_id, docs in tqdm(self.cluster2docs.items()):
            for doc in docs:
                self.doc2cluster[doc] = cluster_id
        pickle_dump(self.doc2cluster, f"{self.output_file}/doc2cluster.pk")

    def build_cluster2length(self):
        length_list = []
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
        self.cluster2length = {}
        for cluster_id, docs in tqdm(self.cluster2docs.items()):
            self.cluster2length[cluster_id] = len(docs)
            length_list.append(len(docs))
        print(f"average length: {sum(length_list)/len(length_list)}")

    def merge(self):
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
        self.doc2cluster = pickle_load(f"{self.output_file}/doc2cluster.pk")
        # data_stats(self.cluster2docs)

        merged_clusters_num = 0
        for cluster, cluster_docs in tqdm(self.cluster2docs.copy().items()):
            if len(cluster_docs) < self.cluster_size:
                merged_clusters_num += 1
                # print(merged_clusters_num)
                for doc in cluster_docs:
                    knns, relative_id = self.load_corresponding_knns(doc, self.num_docs_per_file)
                    top1k, top1k_cluster = self.output_first_doc_knn_not_in_the_cluster(knns[relative_id, :], cluster)
                    # bp()
                    k_cluster_docs = self.cluster2docs[top1k_cluster]

                    if top1k is None or top1k not in k_cluster_docs:
                        continue

                    k_cluster_docs.insert(k_cluster_docs.index(top1k), doc)

                    # update the cluster
                    self.cluster2docs[top1k_cluster] = k_cluster_docs
                    self.doc2cluster[doc] = top1k_cluster
                del self.cluster2docs[cluster]
        print(merged_clusters_num)
        pickle_dump(self.cluster2docs, f"{self.output_file}/cluster2docs_merge.pk")
        
    def analyze_data(self):
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs_merge.pk")
        data_stats(self.cluster2docs)
        
    def output_first_doc_knn(self, knn):
        for k in knn[1:10]:
            if k not in self.seen_docs:
                return k
        return None

    def output_first_doc_knn_not_in_the_cluster(self, knn, cluster_id):
        for k in knn[1:10]:
            k_cluster = self.doc2cluster[k]
            # bp()
            while k_cluster != cluster_id:
                return k, k_cluster
        return None, None

    def write_docs(self):
        sort_doc = self.cluster2list()
        output_folder = f"{self.output_file}/data"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Determine the number of processes to use
        num_processes = 64
        chunks = self.divide_into_chunks(sort_doc, num_processes)
        
        # Create a pool of workers and distribute the work
        args_list = [] 
        for i, chunk in enumerate(chunks):
            args_list.append((chunk, i))

        print(f"data ready: ", len(args_list))
        with multiprocessing.Pool(processes=num_processes) as pool:
            for _ in tqdm(pool.imap(self.write_docs_wrapper, args_list), total=len(args_list)):
                pass

    def write_concat_docs(self):
        sort_doc = self.cluster2independent_list()
        output_folder = f"{self.output_file}/data"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Determine the number of processes to use
        num_processes = 64
        chunks = self.divide_into_chunks(sort_doc, num_processes)
        
        # Create a pool of workers and distribute the work
        args_list = [] 
        for i, chunk in enumerate(chunks):
            args_list.append((chunk, i))

        print(f"data ready: ", len(args_list))
        with multiprocessing.Pool(processes=num_processes) as pool:
            for _ in tqdm(pool.imap(self.write_concat_docs_wrapper, args_list), total=len(args_list)):
                pass

    def divide_into_chunks(self, lst, n):
        """Divide a list into n chunks."""
        # each bucket size is len(lst)/n
        batch_size = len(lst) // n
        for i in tqdm(range(0, len(lst), batch_size)):
            yield lst[i:i + batch_size]

    def write_docs_wrapper(self, args):
        return self.write_docs_single(*args)
    
    def write_concat_docs_wrapper(self, args):
        return self.write_concat_docs_single(*args)

    def write_docs_single(self, sort_doc_chunk, file_index):
        output_folder = f"{self.output_file}/data"
        prev_doc = None
        filter_docs = []
        with open(f"{output_folder}/train_{file_index}.jsonl", "w") as f:
            for doc_id in tqdm(sort_doc_chunk):
                doc = get_document_at_position(self.text_files, self.id2file_pos, doc_id)
                if prev_doc is not None:
                    try:
                        doc_sim = ngram_similarity(doc[self.text_key][:100], prev_doc[self.text_key][:100], self.n)
                    except:
                        prev_doc = doc
                        continue

                    if doc_sim > self.doc_sim_threshold:
                        filter_docs.append(self.cur_k)
                        continue
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                prev_doc = doc
        print(f"filter docs: {len(filter_docs)}")

    def write_concat_docs_single(self, doc_path_chunk, file_index):
        output_folder = f"{self.output_file}/data"
        prev_doc = None
        filter_docs = []
        with open(f"{output_folder}/train_{file_index}.jsonl", "w") as f:
            for doc_path in tqdm(doc_path_chunk):
                doc_id_path = []
                doc_content_path = []
                if len(doc_path) == 0:
                    continue

                for doc_id in doc_path:
                    doc = get_document_at_position(self.text_files, self.id2file_pos, doc_id)
                    if prev_doc is not None:
                        try:
                            doc_sim = ngram_similarity(doc[self.text_key][:100], prev_doc[self.text_key][:100], self.n)
                        except:
                            prev_doc = doc
                            continue

                        if doc_sim > self.doc_sim_threshold:
                            filter_docs.append(doc_id)
                            continue

                    doc_content_path.append(doc[self.text_key])
                    doc_id_path.append(doc['id'])
                    prev_doc = doc

                f.write(json.dumps({
                    'id_path': doc_id_path,
                    'content_path': doc_content_path
                }, ensure_ascii=False) + "\n")

    def cluster2list(self):
        if self.merge_path:
            self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs_merge.pk")
        else:
            self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")

        sort_doc = []
        for _, docs in tqdm(self.cluster2docs.items()):
            sort_doc.extend(docs)
        return sort_doc
    
    def cluster2independent_list(self):
        if self.merge_path:
            self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs_merge.pk")
        else:
            self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")

        sort_doc = []
        for _, docs in tqdm(self.cluster2docs.items()):
            if len(docs) == 0:
                continue
            sort_doc.append(docs)
        return sort_doc
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="en-paper-arxiv")
    parser.add_argument("--output_dir", type=str, default="local/path")
    parser.add_argument("--knn_dir", type=str, default="knn/path")
    parser.add_argument("--text_dir", type=str, default="text/path")
    parser.add_argument("--faiss_config", type=str, default="faiss_config/file")
    parser.add_argument("--process_num", type=int, default=64)

    args = parser.parse_args()
    domain = args.domain
    text_dir = args.text_dir
    process_num = args.process_num

    faiss_cfg = load_config(args.faiss_config)
    text_files = get_text_files(faiss_cfg, text_dir, domain)

    # output dir
    output_dir = f"{args.output_dir}/{domain}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load knn
    knn_dir = f"{args.knn_dir}/knn"

    file_id_text2count = check_doc_line_count(output_dir, text_files, 'circle')
    # build id2file_pos
    if os.path.exists(f"{output_dir}/circle_id2file_pos.pkl"):
        id2file_pos = pickle_load(f"{output_dir}/circle_id2file_pos.pkl")
    else:
        id2file_pos = build_index_all_file_new(text_files, output_dir, 'circle', file_id_text2count)

    faiss_hyper = dict(
        ivf=1500,
        pq=256,
        nprob=64
    )
    merge_path = False # default: True
    sort_member = sort_class(id2file_pos, output_dir, 4096, "content", text_files, knn_dir, faiss_hyper, merge_path)

    sort_member.load_all_knns()
    sort_member.sort()
    sort_member.merge()
    sort_member.analyze_data()
    sort_member.write_docs()
    sort_member.write_concat_docs()
    






    
