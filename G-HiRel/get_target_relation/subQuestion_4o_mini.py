#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Software: PyCharm
import os
import re
import time
import warnings
import openai
import torch

import sys
import pandas as pd


def get_reference_from_txt(reference_data_path):
    reference_data = []
    pattern = re.compile(r'^(\d+)\.\s+sub-question:(.*)')
    with open(reference_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                temp = match.group(2).split(";")
                for i in range(len(temp)):
                    temp[i] = temp[i].lower().strip()

                reference_data.append(temp)

    return reference_data


def prompt_test(init_prompt, sentence, max_retries=5, delay_seconds=3):
    answers = []
    prompts = []
    for index, item in enumerate(sentence):

        prompt = init_prompt

        prompt += f"\nQuestion {index} sentence: {item}" \
                  f"\nQuestion {index}: Based on <Relationships>, what is the target relationship of <Question {index} sentence> ?" \
                  f"\nAnswer {index}:"
        prompts.append(prompt)
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

        answer_temp = response['choices'][0]['message']
        if answer_temp.get('content') is not None:
            answer = response['choices'][0]['message']['content']
            time.sleep(0.1)

            answer_dict = {index: answer}

            if answer_dict.get(index) is not None:
                answers.append(answer_dict.get(index))
            else:
                answers.append("None")
        else:
            answers.append("None")

    return answers


def merget_string(str_list):
    str_result = ""
    for index, str_item in enumerate(str_list):
        if (index) == 0:
            str_result = str_item.replace("\"","")
        else:
            str_result = str_result + "; " + str_item.replace("\"","")
    return str_result


def save_strings_to_file(string_list, file_path):
    with open(file_path, 'w') as file:
        for string in string_list:
            file.write(string + '\n')


def get_target_relations(init_prompt, reference_data, file_output_path):
    target_relations_str = []
    start_time = time.time()
    for index, item in enumerate(reference_data):
        target_relations = prompt_test(init_prompt, item)
        target_relations_str_cur = merget_string(target_relations)
        print(f"{index}. target_relation: {target_relations_str_cur}")
        print("- "*80)

        target_relations_str.append(target_relations_str_cur)

        if ((index + 1) % 10 == 0):
            print(f"finished {index + 1}, time cost: {(time.time() - start_time):.2f}")
            print("- "*80)

    save_strings_to_file(target_relations_str, file_output_path)


def get_init_prompt(init_prompt_path, pid_path):
    with open(init_prompt_path, 'r', encoding='utf-8') as file:
        init_prompt = file.read()

    pid2Name = pd.read_csv(pid_path, encoding="utf-8")
    labels = [f'"{label}"' for label in pid2Name["propertyLabel"]]
    result_string = "{" + ", ".join(labels) + "}"
    init_prompt = init_prompt.replace("<relationships_to_fill>", result_string)
    return init_prompt


if __name__ == "__main__":
    init_prompt_path = r"./prompt/prompt_v1.txt"

    # reference_data_path = r"../get_subQuestion/subQuestion/4o_mini/mquake_subQuestion.txt"
    # file_output_path = r"./target_relation/subQuestion/target_relations_MQuAKE_4o-mini.txt"

    reference_data_path = r"../get_subQuestion/subQuestion/mquake_t_subQuestion.txt"
    file_output_path = r"./target_relation/subQuestion/target_relations_MQuAKE_T_4o-mini.txt"

    pid_path = r"../KG/relationships/pid2Name/pid2Name.csv"

    reference_data = get_reference_from_txt(reference_data_path)
    print(len(reference_data))


    # Set OpenAI key
    openai.api_key = "APIKEY"

    print(init_prompt_path)
    print(reference_data_path)
    print(file_output_path)
    print(pid_path)

    init_prompt = get_init_prompt(init_prompt_path, pid_path)

    get_target_relations(init_prompt, reference_data, file_output_path)
