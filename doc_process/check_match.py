import numpy as np
import os
import json
from ipdb import set_trace as bp
from tqdm import tqdm
import pickle
import statistics
import subprocess
from multiprocessing import Pool
import multiprocessing as mp


def load_knn(file_path):
    knns = np.load(file_path, mmap_mode="r")
    return knns

def build_index(text_file, id2file_pos, file_id_text, doc_id):
    with open(text_file, 'r') as file:
        position = file.tell()
        line = file.readline()
        # with tqdm(total=1000000) as pbar:
        while line:
            id2file_pos[doc_id] = [file_id_text, position]
            # bp()
            doc_id += 1
            position = file.tell()
            line = file.readline()
            # pbar.update(1)
            # tqdm.update(1)
    return id2file_pos

def get_doc_id(file_id, file_id_text2count):
    doc_id = 0
    for i in range(file_id):
        doc_id += file_id_text2count[i]
    return doc_id

def build_index_wrapper(args):
    """
    Wrapper function for build_index to pass multiple arguments.
    """
    return build_index(*args)

def build_index_all_file(text_files, output_index_dir, mode, file_id_text2count):
    assert mode in ['source', 'target', 'circle']
    if not os.path.exists(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl")):
        args_list = []
        files = text_files

        for file_id, text_file in enumerate(files):
        # for file_id in file_id_text2count.keys():
            doc_id = get_doc_id(file_id, file_id_text2count)
            args_list.append((text_file, {}, file_id, doc_id))

        cpu_count = mp.cpu_count()
        print(f"Number of files to process: {len(args_list)} with {cpu_count} CPU!")
        # Using multiprocessing Pool
        with Pool(len(file_id_text2count)) as pool:
            results = list(tqdm(pool.imap(build_index_wrapper, args_list), total=len(args_list)))

        # Merging results and dumping
        id2file_pos = {}
        for i, result in enumerate(results):
            id2file_pos.update(result)

        with open(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl"), "wb") as f:
            pickle.dump(id2file_pos, f)
    else:
        with open(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl"), "rb") as f:
            id2file_pos = pickle.load(f)
    return id2file_pos


def build_index_all_file_new(text_files, output_index_dir, mode, file_id_text2count):
    assert mode in ['source', 'target', 'circle']
    if not os.path.exists(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl")):
        args_list = []
        files = text_files

        for file_id, text_file in enumerate(files):
        # for file_id in file_id_text2count.keys():
            doc_id = get_doc_id(file_id, file_id_text2count)
            args_list.append((text_file, {}, file_id, doc_id))

        cpu_count = mp.cpu_count()
        print(f"Number of files to process: {len(args_list)} with {cpu_count} CPU!")
        # Using multiprocessing Pool
        args_list_bs = len(args_list)
        if len(args_list) > cpu_count:
            args_list_bs = cpu_count
        
        results = []
        for i in range(0, len(args_list), args_list_bs):
            sub_args_list = args_list[i:i+args_list_bs]
            print(f'Processing {len(sub_args_list)} files...')
            with Pool(len(sub_args_list)) as pool:
                sub_results = list(tqdm(pool.imap(build_index_wrapper, sub_args_list), total=len(sub_args_list)))
                results.extend(sub_results)
            print(f'Processed {len(sub_args_list)} files!')
                
        # Merging results and dumping
        id2file_pos = {}
        for i, result in enumerate(results):
            id2file_pos.update(result)

        with open(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl"), "wb") as f:
            pickle.dump(id2file_pos, f)
    else:
        with open(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl"), "rb") as f:
            id2file_pos = pickle.load(f)
    return id2file_pos


def build_index_all_file_norm(text_files, output_index_dir, mode, file_id_text2count):
    assert mode in ['source', 'target', 'circle']
    if not os.path.exists(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl")):
        args_list = []
        files = text_files

        for file_id, text_file in enumerate(files):
        # for file_id in file_id_text2count.keys():
            doc_id = get_doc_id(file_id, file_id_text2count)
            args_list.append((text_file, {}, file_id, doc_id))

        cpu_count = mp.cpu_count()
        print(f"Number of files to process: {len(args_list)} with {cpu_count} CPU!")
        # Using multiprocessing Pool
        
        results = []
        for i in tqdm(range(0, len(args_list))):
            sub_results = build_index_wrapper(args_list[i])
            results.extend(sub_results)
                
        # Merging results and dumping
        id2file_pos = {}
        for i, result in enumerate(results):
            id2file_pos.update(result)

        with open(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl"), "wb") as f:
            pickle.dump(id2file_pos, f)
    else:
        with open(os.path.join(output_index_dir, f"{mode}_id2file_pos.pkl"), "rb") as f:
            id2file_pos = pickle.load(f)
    return id2file_pos


def format_number(number):
    """
    Formats a given number. If it's a single digit (less than 10), it prefixes a zero.
    Otherwise, it returns the number as a string.
    """
    if 0 <= number < 10:
        return f'0{number}'
    else:
        return str(number)

def check_doc_line_count(output_index_dir, text_files, mode):
    if os.path.exists(f"{output_index_dir}/{mode}_file_id_text2count.pkl"):
        file_id_text2count = pickle_load(f"{output_index_dir}/{mode}_file_id_text2count.pkl")
    else:
        file_id_text2count = {}
        files = text_files
        for file_id, text_file in tqdm(enumerate(files)):
            try:
                # Using wc -l command to count lines
                result = subprocess.run(['wc', '-l', text_file], capture_output=True, text=True)
                if result.returncode == 0:
                    line_count = int(result.stdout.split()[0])
                    file_id_text2count[file_id] = line_count
                    print(f"File {text_file} has {line_count} lines.")
                else:
                    print(f"Error in processing file {text_file}: {result.stderr}")
            except Exception as e:
                print(f"An error occurred while processing {text_file}: {e}")
        pickle_dump(file_id_text2count, f"{output_index_dir}/{mode}_file_id_text2count.pkl")
    return file_id_text2count


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def get_document_at_position(text_files, id2file_pos, id):
    files = text_files
    file_id, position = id2file_pos[id]
    jsonl_file_path = files[int(file_id)]

    with open(jsonl_file_path, 'r') as file:
        # print(position)
        file.seek(position)
        line = file.readline()
        return json.loads(line)

def data_stats(clusterid2docids):
    clusterid2count  = {}
    total_docs = 0
    cluster_21 = 0
    for k, v in tqdm(clusterid2docids.items()):
        clusterid2count[k] = len(v)
        total_docs += len(v)
        if len(v) >= 21:
            cluster_21 += 1
    count_list = list(clusterid2count.values())
    q = statistics.quantiles(count_list, n=100)
    print(f"quantiles 25%: {q[25]}, 50%: {q[50]}, 75%: {q[75]}")
    print(f"total_docs: ", total_docs)
    # number of clusters with more than 21 docs:
    useful_cluster = len([i for i in count_list if i >= 21])
    # bp()
    print(f"number of clusters with more than 21 docs: ", useful_cluster)
