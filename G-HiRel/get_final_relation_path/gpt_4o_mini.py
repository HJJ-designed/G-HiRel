#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Software: PyCharm
import warnings
import os
import time
import ijson
import openai
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def group_data_by_id(result):
    group_dict = defaultdict(lambda: [[], [], []])
    for item in result:
        try:
            id_str, content = item.split(". ", 1)
            id_int = int(id_str)
            group_id = id_int // 3
            position_in_group = id_int % 3
            group_dict[group_id][position_in_group].append(content)
        except ValueError:
            print(f"Skipping invalid row: {item}")
    return group_dict


def formed_relation_path_str(reasoning_path, pid2Relation, qid2Name):
    relation_reasoning_path = set()

    for path in reasoning_path:
        cur_relation_path = ""

        triples = path.split(";")
        for triple_index in range(len(triples)):
            triple = triples[triple_index].split(",")
            if triple_index == 0:
                cur_relation_path = qid2Name.get(triple[0]) + " -> " + pid2Relation.get(triple[1])
            else:
                cur_relation_path = cur_relation_path + " -> " + pid2Relation.get(triple[1])

        relation_reasoning_path.add(cur_relation_path)

    return list(relation_reasoning_path)


def get_target_relations_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    target_relations = {}
    for index, line in enumerate(lines):
        lines[index] = lines[index].split(".")[-1].strip()
        target_relations[index] = [item.strip() for item in lines[index].split(";")]

    return target_relations


def process_case_prompt(question_num, question, reasoning_path, prompt):
    temp_prompt = prompt + f"\nQuestion_{question_num}: " + question
    temp_prompt = temp_prompt + f"\nQuestion_{question_num} Information:"

    for reasoning_path_index in range(len(reasoning_path)):
        temp_prompt = temp_prompt + f"\n({reasoning_path_index + 1}) " + reasoning_path[reasoning_path_index]

    temp_prompt = temp_prompt + f"\nQuestion_{question_num} relation reasoning path:"

    return temp_prompt


def get_answer_prompts(prompt, question_id, max_retries = 5, delay_seconds = 3):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    response = None
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages
            )
            break
        except (openai.error.Timeout,
                openai.error.APIError,
                openai.error.APIConnectionError,
                openai.error.RateLimitError,
                openai.error.InvalidRequestError,
                openai.error.PermissionError
                ) as e:
            print(f"Request attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying after {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print("Maximum number of retries reached. Unable to retrieve GPT results.")
                raise e  # Raise the exception if all retries have been exhausted

    if not response:
        raise RuntimeError("Failed to call GPT: no available response.")


    if response['choices'][0]['message'].get('content') is not None:
        answer = response['choices'][0]['message']['content']
    else:
        answer = "None"
    time.sleep(0.1)
    answer = answer.split("End of answer.",1)[0]
    print(f"Question_{question_id} relation reasoning path:",answer.strip())


if __name__ == "__main__":
    data_start_time = time.time()

    openai.api_key = "APIKEY"

    pid2Relation_path = r"../KG/relationships/pid2Name/pid2Name.csv"
    init_promptes_path = r"./prompts/few_shot_prompt_v1.txt"

    result_path = f"../get_subGraph/results/MQuAKE-CF-3k/reasoning_result_edit_1_topK_3.csv"
    start_entity_path = r"../KG/start_entities/mquake_entity2qids.csv"
    MQUAKE_path = r"../data/MQUAKE/MQuAKE-CF-3k.json"
    target_relation_path = r"../get_target_relation/target_relation/subQuery/target_relations_MQuAKE_gpt_4o_mini.txt"


    print("pid2Relation_path", pid2Relation_path)
    print("init_promptes_path", init_promptes_path)
    print("result_path", result_path)
    print("start_entity_path", start_entity_path)
    print("MQUAKE_path", MQUAKE_path)
    print("target_relation_path", target_relation_path)

    pid2Relation = pd.read_csv(pid2Relation_path, encoding="UTF-8")
    pid2Relation = pid2Relation.values
    pid2Relation_dict = {row[0]: row[1] for row in pid2Relation}

    start_entity = pd.read_csv(start_entity_path, header=None, encoding="UTF-8").fillna("None")
    start_entity = start_entity.values.astype(str)

    qid2Name = {}

    for item in start_entity:
        name = item[0]
        qids = item[1]
        if ";" in qids:
            qids = qids.split(";")
            for qid in qids:
                qid2Name[qid] = name
        else:
            qid2Name[qids] = name

    target_relations = get_target_relations_from_txt(target_relation_path)
    print(len(target_relations))

    print(f"Reference data is complete. Time cost: {(time.time() - data_start_time):.2f} seconds. Starting to construct the prompt...")
    result = pd.read_csv(result_path, header=None, encoding="UTF-8")[0].tolist()

    group_dict = group_data_by_id(result)
    for group_dict_key in group_dict.keys():
        group_dict_value = []
        for item in group_dict.get(group_dict_key):
            item_relation_path = formed_relation_path_str(item, pid2Relation_dict, qid2Name)
            group_dict_value.append(item_relation_path)
        group_dict[group_dict_key] = group_dict_value

    with open(init_promptes_path, 'r', encoding='utf-8') as file:
        init_prompt = file.read()

    start_time = time.time()
    with open(MQUAKE_path, 'rb') as f:
        mquake_objects = ijson.items(f, 'item')
        for case_id, mquake_object in enumerate(mquake_objects):
            reasoning_path_case = group_dict.get(case_id)

            if reasoning_path_case is None:
                continue

            for question_index in range(3):
                if question_index >= len(reasoning_path_case):
                    continue

                reasoning_path_cur = reasoning_path_case[question_index]
                target_relation_cur = target_relations.get(case_id * 3 + question_index)

                candidate_group = reasoning_path_cur
                relation_num = 1
                while (len(candidate_group) > 1 and relation_num <= len(target_relation_cur)):
                    cur_candidate = []

                    for reasoning_path_item in candidate_group:
                        reasoning_final_relation = reasoning_path_item.split("->")[-relation_num].strip()
                        if target_relation_cur[-relation_num] == reasoning_final_relation:
                            cur_candidate.append(reasoning_path_item)
                    candidate_group = cur_candidate

                    relation_num += 1

                if len(candidate_group) == 1:
                    print(f"Question_{case_id * 3 + question_index} relation reasoning path: {candidate_group[0]}")
                else:
                    cur_prompt = process_case_prompt(case_id * 3 + question_index,
                                                     mquake_object.get("questions")[question_index],
                                                     reasoning_path_cur,
                                                     init_prompt)
                    get_answer_prompts(cur_prompt, case_id * 3 + question_index)
    print(f"total time cost: {(time.time() - start_time)}")