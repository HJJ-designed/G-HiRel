#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Software: PyCharm
import csv
import gc
import os
import re
import time
import warnings

import ijson
import openai
import pandas as pd
from collections import defaultdict
import sys
import io
import requests


def extract_and_process_answers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    relation_pattern = re.compile(r'Question_(\d+) relation reasoning path: (.+)')
    answers = re.findall(relation_pattern, text)

    result = {}
    for answer in answers:
        question_id = int(answer[0])
        reasoning_path = answer[1]
        if question_id not in result.keys():
            result[question_id] = reasoning_path
    return result


def get_relation_path(content, qid2Name, pid2Name):
    relation_path = ""
    triples = content.split(";")
    for i in range(len(triples)):
        triple = triples[i].split(",")
        if i == 0:
            relation_path = qid2Name.get(triple[0], "EmpthEntityName") + " -> " + pid2Name.get(triple[1], "EmptyRelationName")
        else:
            relation_path = relation_path + " -> " + pid2Name.get(triple[1], "EmptyRelationName")
    return relation_path


def get_reasoning_path_str(reasoning_path):
    triples = reasoning_path.split(";")
    for index in range(len(triples)):
        triples[index] = triples[index].split(",")

    reasoning_path_str = ""
    for index in range(len(triples)):
        if index == 0:
            reasoning_path_str = qid2Name.get(triples[index][0], "EmpthEntityName") + \
                                 " -> " + pid2Name.get(triples[index][1], "EmptyRelationName") + \
                                 " -> " + qid2Name.get(triples[index][2], "EmpthEntityName")
        else:
            reasoning_path_str = reasoning_path_str + "; " + qid2Name.get(triples[index][0], "EmpthEntityName") + \
                                 " -> " + pid2Name.get(triples[index][1], "EmptyRelationName") + \
                                 " -> " + qid2Name.get(triples[index][2], "EmpthEntityName")
    return reasoning_path_str


def get_triple_str(triple):
    return triple[0] + "," + triple[1] + "," + triple[2]


def merge_answers_by_key(*file_paths):
    merged_answers = {}
    for file_path in file_paths:
        answers = extract_and_process_answers(file_path)
        for key, value in answers.items():
            merged_answers[key] = value
    return merged_answers


def build_prompt(question_id, question, reasoning_paths, edit_triples, prompt):
    reasoning_text = ""
    if len(reasoning_paths) == 0:
        reasoning_text += f"(1) None\n"
    else:
        for i, path in enumerate(reasoning_paths):
            reasoning_text += f"({i + 1}) {get_reasoning_path_str(path)}\n"

    cur_edit = []
    triples_cur_path = []
    for path in reasoning_paths:
        triples_cur_path += path.split(";")

    triples_cur_path = list(set(triples_cur_path))
    for triple in edit_triples:
        if triple in triples_cur_path:
            cur_edit.append(get_reasoning_path_str(triple))

    edit_text = ""
    if len(cur_edit) == 0:
        edit_text += f"(1) None\n"
    else:
        for i, edit in enumerate(cur_edit):
            edit_text += f"({i + 1}) {edit}\n"

    cur_prompt = prompt + f"\nQuestion_{question_id}: {question}"
    cur_prompt = cur_prompt + f"\nQuestion_{question_id} edit information:\n{edit_text}"
    cur_prompt = cur_prompt + f"Question_{question_id} candidate reasoning path:\n{reasoning_text}"
    cur_prompt = cur_prompt + f"please select one reasoning path from Question_{question_id} candidate reasoning path to answer Question_{question_id}.\n"
    cur_prompt = cur_prompt + f"Question_{question_id} answer:"

    return cur_prompt


def build_prompt_v1(question_id, question, reasoning_paths, edit_triples, prompt):
    reasoning_text = ""
    if len(reasoning_paths) == 0:
        reasoning_text += f"(1) None\n"
    else:
        for i, path in enumerate(reasoning_paths):
            reasoning_text += f"({i + 1}) {path}\n"

    cur_edit = []
    triples_cur_path = []
    for path in reasoning_paths:
        triples_cur_path += path.split("; ")

    triples_cur_path = list(set(triples_cur_path))
    for triple in edit_triples:
        triple = get_reasoning_path_str(triple)
        if triple in triples_cur_path:
            cur_edit.append(triple)

    edit_text = ""
    if len(cur_edit) == 0:
        edit_text += f"(1) None\n"
    else:
        for i, edit in enumerate(cur_edit):
            edit_text += f"({i + 1}) {edit}\n"

    cur_prompt = prompt + f"\nQuestion_{question_id}: {question}"
    cur_prompt = cur_prompt + f"\nQuestion_{question_id} edit information:\n{edit_text}"
    cur_prompt = cur_prompt + f"Question_{question_id} candidate reasoning path:\n{reasoning_text}"
    cur_prompt = cur_prompt + f"please select one reasoning path from Question_{question_id} candidate reasoning path to answer Question_{question_id}.\n"
    cur_prompt = cur_prompt + f"Question_{question_id} answer:"
    return cur_prompt


def generate_prompts_for_question(question_id, question_text, reasoning_paths, edit_triples,
                                  prompt1, prompt2, prompt3, batch_nums=5):
    num_paths = len(reasoning_paths)
    prompts = []

    if num_paths == 0:
        prompt = build_prompt(question_id, question_text, [], edit_triples, prompt1)
        prompts.append(prompt)
    elif num_paths == 1:
        prompt = build_prompt(question_id, question_text, reasoning_paths, edit_triples, prompt2)
        prompts.append(prompt)
    elif 2 <= num_paths <= batch_nums:
        prompt = build_prompt(question_id, question_text, reasoning_paths, edit_triples, prompt3)
        prompts.append(prompt)
    else:
        num_chunks = (num_paths + batch_nums - 1) // batch_nums
        for batch_i in range(num_chunks):
            chunk_paths = reasoning_paths[batch_i * batch_nums:(batch_i + 1) * batch_nums]
            prompt = build_prompt(str(question_id) + f"_{batch_i}", question_text, chunk_paths, edit_triples, prompt3)
            prompts.append(prompt)

    return prompts


def generate_prompts_for_question_v1(question_id, question_text, reasoning_paths, edit_triples,
                                     prompt1, prompt2, prompt3, count=0, batch_nums=5):
    num_paths = len(reasoning_paths)
    prompts = []

    if num_paths == 0:
        prompt = build_prompt_v1(question_id, question_text, [], edit_triples, prompt1)
        prompts.append(prompt)
    elif num_paths == 1:
        prompt = build_prompt_v1(question_id, question_text, reasoning_paths, edit_triples, prompt2)
        prompts.append(prompt)
    elif 2 <= num_paths <= batch_nums:
        prompt = build_prompt_v1(question_id, question_text, reasoning_paths, edit_triples, prompt3)
        prompts.append(prompt)
    else:
        num_chunks = (num_paths + batch_nums - 1) // batch_nums
        for batch_i in range(num_chunks):
            chunk_paths = reasoning_paths[batch_i * batch_nums:(batch_i + 1) * batch_nums]
            prompt = build_prompt_v1(str(question_id) + f"_{count + batch_i}", question_text, chunk_paths, edit_triples,
                                     prompt3)
            prompts.append(prompt)

    return prompts


def get_answer(prompts, max_retries = 5, delay_seconds = 3):
    start_time = time.time()

    answer_list = []
    for prompt in prompts:
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
        print_prompt = prompt.split("Now is start, please answer the question as the examples and do not output any extra information.")[-1].strip()
        print(print_prompt)

        print(answer.strip())
        print("- "*80)


        if "{" not in answer:
            answer_path = "None"
        else:
            answer_path = answer.split("{")[-1].strip()
            answer_path = answer_path.split("}", 1)[0].strip()
        answer_list.append(answer_path)

    print(f"{time.time() - start_time}")
    return answer_list

def write_to_csv(elements, output_path):
    with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for select_path in elements:
            if isinstance(select_path, str):
                csv_writer.writerow([select_path])
            else:
                csv_writer.writerow(select_path)


if __name__ == "__main__":

    openai.api_key = "APIKEY"
    edit = 1
    topK = 3

    file_path = f'../get_final_relation_path/final_relation_path/MQuAKE-CF-3k/edit_{edit}_topK_{topK}.txt'
    file2_path = f"../get_subGraph/results/MQuAKE-CF-3k/reasoning_result_edit_{edit}_topK_{topK}.csv"
    output_path = f"./results/MQuAKE-CF-3k/edit_{edit}_topK_{topK}.csv"

    qid2Name_path = r"../data/wikidata/items_to_label_and_alias.csv"
    MQUAKE_path = r"../data/MQUAKE/MQuAKE-CF-3k.json"
    pid2Name_path = r"../KG/relationships/pid2Name/pid2Name.csv"

    promt1_path = r"./prompt/prompt1.txt"
    promt2_path = r"./prompt/prompt2.txt"
    promt3_path = r"prompt/prompt3.txt"

    print("file_path: ", file_path)
    print("file2_path: ", file2_path)
    print("output_path: ", output_path)
    print("qid2Name_path: ", qid2Name_path)
    print("MQUAKE_path: ", MQUAKE_path)
    print("pid2Name_path: ", pid2Name_path)
    print("promt1_path: ", promt1_path)
    print("promt2_path: ", promt2_path)
    print("promt3_path: ", promt3_path)

    with open(promt1_path, 'r', encoding='utf-8') as file:
        promt1 = file.read()

    with open(promt2_path, 'r', encoding='utf-8') as file:
        promt2 = file.read()

    with open(promt3_path, 'r', encoding='utf-8') as file:
        promt3 = file.read()

    start_entity = pd.read_csv(qid2Name_path, encoding="UTF-8").fillna("None")
    qid2Name = dict(zip(start_entity['Qid'], start_entity['name']))

    pid2Name_data = pd.read_csv(pid2Name_path, encoding="UTF-8")
    pid2Name = {row[0]: row[1] for row in pid2Name_data.values}

    answers = merge_answers_by_key(file_path)
    print("len_answers:", len(answers))

    result = pd.read_csv(file2_path, header=None, encoding="UTF-8")[0].tolist()
    result_dict = defaultdict(list)
    qid_result_dict = defaultdict(list)

    nums = 0
    for item in result:
        try:
            id_str, content = item.split(". ", 1)
            id_int = int(id_str)
            relation_path = get_relation_path(content, qid2Name, pid2Name)
            reference_relation_path = answers.get(id_int)
            if reference_relation_path is not None:
                reference_relation_path = reference_relation_path.strip()

            if reference_relation_path is not None and relation_path.lower() == reference_relation_path.lower():
                result_dict[id_int].append(content)
                nums += 1
        except ValueError:
            print(f"Skipped invalid row: {item}")

    edit_triples_batch = []
    edit_triples = []
    with open(MQUAKE_path, 'rb') as f:
        mquake_objects = ijson.items(f, 'item')
        for i, mquake_object in enumerate(mquake_objects):
            for triple in mquake_object.get("orig").get("edit_triples"):
                triple = get_triple_str(triple)
                if triple not in edit_triples:
                    edit_triples.append(triple)

            if (i + 1) % edit == 0:
                edit_triples_batch.append(edit_triples)
                edit_triples = []

        if len(edit_triples) > 0:
            edit_triples_batch.append(edit_triples)
            edit_triples = []

    edit_triples = None
    gc.collect()

    start_time = time.time()
    with open(MQUAKE_path, 'rb') as f:
        mquake_objects = ijson.items(f, 'item')

        answer_path_list = []
        for case_id, mquake_object in enumerate(mquake_objects):
            questions = mquake_object.get("questions")
            batch_index = int(case_id / edit)
            edit_triples = edit_triples_batch[batch_index]

            for question_i, question in enumerate(questions):
                question_id = case_id * 3 + question_i

                reasoning_path = result_dict.get(case_id * 3 + question_i)

                if reasoning_path is None:
                    reasoning_path = []

                reasoning_path = list(set(reasoning_path))
                prompts1 = generate_prompts_for_question(question_id, question, reasoning_path, edit_triples, promt1,
                                                         promt2, promt3)

                answer_list = get_answer(prompts1)
                answer_list = list(set(answer_list))
                answer_list = [ans for ans in answer_list if ans not in (None, 'None')]

                answer_count = 0
                if len(answer_list) == 1:
                    answer_path_list.append(f"{question_id}. {answer_list[0]}")
                else:
                    for ans_id, ans_path in enumerate(answer_list):
                        answer_path_list.append(f"{question_id}_{ans_id}. {ans_path}")
                        answer_count += 1
                while len(answer_list) > 1:
                    prompts2 = generate_prompts_for_question_v1(question_id, question, answer_list, edit_triples,
                                                                promt1, promt2, promt3, answer_count)
                    answer_list = get_answer(prompts2)
                    answer_list = list(set(answer_list))
                    answer_list = [ans for ans in answer_list if ans not in (None, 'None')]

                    if len(answer_list) == 1:
                        answer_path_list.append(f"{question_id}. {answer_list[0]}")
                    else:
                        for ans_id, ans_path in enumerate(answer_list):
                            answer_path_list.append(f"{question_id}_{answer_count}. {ans_path}")
                            answer_count += 1

            if (case_id + 1) % 10 == 0:
                print(f"Completed summarization for {case_id + 1} instances. Time cost: {(time.time() - start_time):.2f} seconds.")
                print("- " * 80)
                write_to_csv(answer_path_list, output_path)
                answer_path_list = []

        if len(answer_path_list) > 0:
            write_to_csv(answer_path_list, output_path)
            answer_path_list = []

        print(f"Completed summarization for {case_id + 1} instances. Time cost: {(time.time() - start_time):.2f} seconds.")
        print("- " * 80)