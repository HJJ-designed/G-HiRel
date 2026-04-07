#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Software: PyCharm

import os
import re
import time

import ijson
import pandas as pd
import igraph as ig
import csv
import numpy as np
import gc
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

import sys

cuda_num = 0
torch.cuda.set_device(cuda_num)


def get_knowledge_graph(data_path):
    df = pd.read_csv(data_path)
    entities = pd.concat([df['subject'], df['object']]).unique().astype(str)
    idx_to_entity = {idx: entity for idx, entity in enumerate(entities)}
    entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}

    relations = df['property'].unique()
    df['subject'] = df['subject'].map(entity_to_idx)
    df['object'] = df['object'].map(entity_to_idx)
    g = ig.Graph(directed=True)
    g.add_vertices(len(idx_to_entity))
    g.vs["name"] = list(idx_to_entity.values())
    edges = list(zip(df['subject'], df['object']))
    g.add_edges(edges)
    g.es['property'] = df['property'].tolist()
    return g, idx_to_entity, entity_to_idx


def get_node_neighbors(graph, idx_to_entity, entity_to_idx, subject_name):
    if subject_name not in entity_to_idx:
        return [], []
    subject_idx = entity_to_idx[subject_name]
    neighbor_indices = graph.neighbors(subject_idx, mode='out')
    neighbors = [idx_to_entity[idx] for idx in neighbor_indices]
    relationships = [
        [idx_to_entity[subject_idx], graph.es[edge_index]['property'], idx_to_entity[target_idx]]
        for edge_index, target_idx in zip(graph.incident(subject_idx, mode='out'), neighbor_indices)
    ]

    return neighbors, relationships


def get_reference_from_txt(reference_data_path):
    reference_data = []
    pattern = re.compile(r'^(\d+)\.\s+Split:(.*)')
    with open(reference_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                temp = match.group(2).split(";")
                for i in range(len(temp)):
                    temp[i] = temp[i].lower().strip()

                reference_data.append(temp)

    return reference_data


def find_edit_triples(mquake_object):
    edit_triples = []
    cur_edit_triples = mquake_object.get("edit_triples")
    edit_triples += cur_edit_triples

    return edit_triples


def get_neighbor_edit_triple_in_place(graph, idx_to_entity, entity_to_idx, subject_name, edit_triples):
    _, relationships = get_node_neighbors(graph, idx_to_entity, entity_to_idx, subject_name)
    relationships_edit = []

    from collections import defaultdict, deque

    original_predicates = defaultdict(deque)
    for triple in relationships:
        subj, pred, obj = triple
        original_predicates[pred].append(triple)

    edit_predicates = defaultdict(deque)
    for triple in edit_triples:
        subj, pred, obj = triple
        if subj == subject_name:
            edit_predicates[pred].append(triple)

    for pred, edit_triples_queue in edit_predicates.items():
        if pred in original_predicates:
            original_triples_queue = original_predicates[pred]
            while edit_triples_queue and original_triples_queue:
                original_triples_queue.popleft()
                new_triple = edit_triples_queue.popleft()
                relationships_edit.append(new_triple)
            while edit_triples_queue:
                new_triple = edit_triples_queue.popleft()
                relationships_edit.append(new_triple)
        else:
            while edit_triples_queue:
                new_triple = edit_triples_queue.popleft()
                relationships_edit.append(new_triple)
    for pred, triples_queue in original_predicates.items():
        while triples_queue:
            relationships_edit.append(triples_queue.popleft())

    return relationships_edit


def get_test_result(question, reasoning_path, classification_model, tokenizer, device):
    classification_input = f"Is the reasoning Path:\" {reasoning_path}\" sufficient to describe the sentence:\"{question}\" ?" \
                           f"Answer only yes or no."
    input_ids = tokenizer.encode(classification_input, return_tensors="pt").to(device)
    classification_model.eval()
    with torch.no_grad():
        outputs = classification_model.generate(input_ids, max_length=3)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def get_reasoning_path_str(start_entity, reasoning_path, pid2Name):
    reasoning_path_str = start_entity

    for item in reasoning_path:
        reasoning_path_str = reasoning_path_str + ", " + str(pid2Name.get(item[1]))
    return reasoning_path_str


def get_cosine_similarity(sentence_embedding, reasoning_embedding):
    return F.cosine_similarity(sentence_embedding, reasoning_embedding, dim=-1)


def get_top_k_indices(cosine_similarities, top_k):
    sorted_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)
    return sorted_indices[:top_k]


def get_reasoning_result_str(index, reasoning_result):
    result_str_list = []
    for item in reasoning_result:
        str_res = str(index) + ". "
        for i in range(len(item)):
            str_res_cur = item[i][0] + "," + item[i][1] + "," + item[i][-1]
            if i == 0:
                str_res = str_res + str_res_cur
            else:
                str_res = str_res + ";" + str_res_cur

        result_str_list.append(str_res)

    return result_str_list

def write_to_csv(elements, output_path):
    with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        for triple in elements:
            csv_writer.writerow(triple)


if __name__ == "__main__":
    torch.cuda.set_device(cuda_num)

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device is: {device}")

    max_reasoning_depth = 2

    top_k = 3
    edit_case_nums = 1
    reasoning_result_out_path = f"./results/MQuAKE-CF-3k/reasoning_result_edit_{edit_case_nums}_topK_{top_k}.csv"
    wikidata_path = r"../data/wikidata/triples.csv"
    print("output_path:", reasoning_result_out_path)
    print("KG path:", wikidata_path)

    MQUAKE_path = r"../data/MQUAKE/MQuAKE-CF-3k.json"
    start_entity_path = r"../KG/start_entities/mquake_entity2qids.csv"
    reference_data_path = r"../QuestionReformulation/sub_query/mquake_sub_query.txt"
    pid2Name_path = r"../KG/relationships/pid2Name/pid2Name.csv"

    print("start_entity_path:", start_entity_path)
    print("reference_data_path:", reference_data_path)
    print("MQUAKE_path:", MQUAKE_path)
    print("pid2Name_path:", pid2Name_path)

    start_entity = pd.read_csv(start_entity_path, header=None, encoding="UTF-8").fillna("None")
    start_entity = start_entity.values.astype(str)
    print(len(start_entity))

    reference_data = get_reference_from_txt(reference_data_path)
    print(len(reference_data))

    pid2Name = pd.read_csv(pid2Name_path, encoding="UTF-8")
    pid2Name = pid2Name.values.astype(str)
    pid2Name = {k: v for k, v in pid2Name}
    print(len(pid2Name.keys()))
    print("reference data has been loaded")

    # Wikidata Knowledge Graph
    start_time = time.time()
    wikidata_graph, idx_to_entity, entity_to_idx = get_knowledge_graph(wikidata_path)
    print(f"knowledge graph has been constructed，time cost：{(time.time() - start_time):.2f} s")
    print("- " * 80)

    model_path = "../plm_model_cache/sentence-transformer/flan-t5-large"
    sentence_transformers_path = r"../plm_model_cache/sentence-transformer/msmarco-MiniLM-L6-v3"
    print(model_path)
    print(sentence_transformers_path)

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    classification_model = T5ForConditionalGeneration.from_pretrained(model_path, ignore_mismatched_sizes=True).to(
        device)

    encoder_model = SentenceTransformer(sentence_transformers_path).to(device)
    print("Model has been prepared.")
    print("- " * 80)

    mquake_data = []
    with open(MQUAKE_path, 'rb') as f:
        mquake_objects = ijson.items(f, 'item')
        for i, mquake_object in enumerate(mquake_objects):
            muqake_dict = {"case_id": mquake_object.get("case_id"),
                           "edit_triples": mquake_object.get("orig").get("edit_triples"),
                           "questions": mquake_object.get("questions")}
            mquake_data.append(muqake_dict)

    reasoning_path_all = []
    reasoning_start_time = time.time()

    edit_triples_batch = []
    edit_triples = []
    for i, mquake_object in enumerate(mquake_data):
        edit_triples_temp = mquake_object.get("edit_triples")
        for triple in edit_triples_temp:
            triple = tuple(triple)
            if triple not in edit_triples:
                edit_triples.append(triple)

        if (i + 1) % edit_case_nums == 0:
            edit_triples = [list(triple) for triple in edit_triples]
            edit_triples_batch.append(edit_triples)
            edit_triples = []
    if len(edit_triples) > 0:
        edit_triples = [list(triple) for triple in edit_triples]
        edit_triples_batch.append(edit_triples)

    edit_triples = None
    gc.collect()
    print("edit_triples_batch nums is:", len(edit_triples_batch))
    print("Editing information has been grouped...")

    for i, mquake_object in enumerate(mquake_data):
        start_time = time.time()

        batch_index = int(i / edit_case_nums)
        edit_triples = edit_triples_batch[batch_index]
        questions = mquake_object.get("questions")
        for questions_id in range(len(questions)):
            cur_reasoning_path = []
            cur_reference_data = reference_data[i * 3 + questions_id]
            cur_reference_data_index = 0

            cur_start_entity = start_entity[i, :]

            if cur_start_entity[0] == "None":
                continue

            reasoning_count = 0
            need_reasoning = True
            while True:
                answers = []
                if len(cur_reasoning_path) == 0:
                    answer = get_test_result(cur_reference_data[cur_reference_data_index], cur_start_entity[0],
                                             classification_model, tokenizer, device)
                    answers.append(answer.lower())
                else:
                    for item in cur_reasoning_path:
                        item_reasoning_path_str = get_reasoning_path_str(cur_start_entity[0], item, pid2Name)
                        answer = get_test_result(cur_reference_data[cur_reference_data_index], item_reasoning_path_str,
                                                 classification_model, tokenizer, device)
                        answers.append(answer.lower())

                for answer in answers:
                    if "yes" in answer:
                        cur_reference_data_index += 1
                        reasoning_count = 0
                        need_reasoning = False
                        break

                if need_reasoning == False:
                    need_reasoning = True
                else:
                    reasoning_count += 1
                    cur_reasoning_path_candidate = []
                    if len(cur_reasoning_path) == 0:
                        if ";" not in cur_start_entity[1]:
                            edited_neighbors = get_neighbor_edit_triple_in_place(wikidata_graph, idx_to_entity,
                                                                                 entity_to_idx,
                                                                                 cur_start_entity[1],
                                                                                 edit_triples)
                            for edited_neighbor in edited_neighbors:
                                cur_reasoning_path_candidate.append([edited_neighbor])
                        else:
                            cur_start_entity_list = cur_start_entity[1].strip().split(";")
                            for item in cur_start_entity_list:
                                edited_neighbors = get_neighbor_edit_triple_in_place(wikidata_graph, idx_to_entity,
                                                                                     entity_to_idx,
                                                                                     item,
                                                                                     edit_triples)

                                for edited_neighbor in edited_neighbors:
                                    cur_reasoning_path_candidate.append([edited_neighbor])
                    else:
                        for item in cur_reasoning_path:
                            edited_neighbors = get_neighbor_edit_triple_in_place(wikidata_graph, idx_to_entity,
                                                                                 entity_to_idx,
                                                                                 item[-1][-1],
                                                                                 edit_triples)

                            for edited_neighbor in edited_neighbors:
                                cur_reasoning_path_candidate.append(item + [edited_neighbor])

                    with torch.no_grad():
                        cur_reference_embedding = encoder_model.encode(
                            cur_reference_data[cur_reference_data_index],
                            convert_to_tensor=True
                        )

                    cur_reasoning_embeddings = []
                    cur_candidate_strs = []
                    for candidate_i in cur_reasoning_path_candidate:
                        candidate_i_str = get_reasoning_path_str(cur_start_entity[0], candidate_i, pid2Name)
                        cur_candidate_strs.append(candidate_i_str)

                    cur_candidate_strs = list(set(cur_candidate_strs))

                    for candidate_i_str in cur_candidate_strs:
                        with torch.no_grad():
                            candidate_i_embedding = encoder_model.encode(
                                candidate_i_str,
                                convert_to_tensor=True
                            )
                        cur_reasoning_embeddings.append(candidate_i_embedding)
                    cosine_smilarities = []
                    for reasoning_embedding in cur_reasoning_embeddings:
                        cosine_smilarities.append(get_cosine_similarity(cur_reference_embedding, reasoning_embedding))
                    top_k_indices = get_top_k_indices(cosine_smilarities, top_k)

                    cur_relation_path = [cur_candidate_strs[candidate_index] for candidate_index in top_k_indices]
                    if len(cur_relation_path) > 0:
                        cur_reasoning_path = []

                        for candidate_i in cur_reasoning_path_candidate:
                            candidate_i_str = get_reasoning_path_str(cur_start_entity[0], candidate_i, pid2Name)
                            if candidate_i_str in cur_relation_path and candidate_i not in cur_reasoning_path:
                                cur_reasoning_path.append(candidate_i)
                    else:
                        reasoning_count = max_reasoning_depth

                if reasoning_count >= max_reasoning_depth:
                    cur_reference_data_index += 1
                    reasoning_count = 0
                    need_reasoning = False

                if cur_reference_data_index == len(cur_reference_data):
                    break
            reasoning_result_str = get_reasoning_result_str(i * 3 + questions_id, cur_reasoning_path)
            reasoning_path_all += reasoning_result_str

        if (i + 1) % 4 == 0:
            reasoning_path_all = np.array(reasoning_path_all).reshape(-1, 1)
            write_to_csv(reasoning_path_all, reasoning_result_out_path)
            reasoning_path_all = []
            print(f"finished {(i + 1) * 3}, time cost: {(time.time() - reasoning_start_time):.2f}")
