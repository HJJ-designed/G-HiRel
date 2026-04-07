#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Software: PyCharm
import os
import csv
import time

import ijson
import openai
import pandas as pd

def append_text_to_file(filename, text):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(text + "\n" + "- " * 40 + "\n")


def get_answer(test_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response['choices'][0]['message']['content']


def question_split_prompt_generate_api(objects,start_entities):
    init_prompt_path = "./prompts/prompt_v1.txt"

    with open(init_prompt_path, 'r', encoding='utf-8') as file:
        init_Prompt = file.read()

    prompt = init_Prompt + "\n"
    question_counts = []

    for i, obj in enumerate(objects):
        questions = obj.get("questions")

        for j in range(len(questions)):
            question_prompt = f"\n\n{i * 3 + 1 + j}. Question+PLM_error: " + questions[j]
            entity_prompt = f"\nStart Entity:{start_entities[i]}"
            prompt = prompt + question_prompt + entity_prompt

        for j in range(len(questions)):
            question_counts.append(i*3+j)

        if len(question_counts) % 3 == 0:
            response = get_answer(prompt)
            print(response)
            print("- " * 40)
            time.sleep(0.1)

            prompt = init_Prompt + "\n"


if __name__=="__main__":
    # Set OpenAI key
    openai.api_key = "APIKEY"

    data_path = r"../data/MQUAKE/MQuAKE-CF-3k.json"
    # data_path = r"../data/MQUAKE-T/MQuAKE-T.json"

    start_entity_path = r"../KG/start_entities/mquake_entity2qids.csv"
    start_entity = pd.read_csv(start_entity_path,header=None,encoding="UTF-8").fillna("None")
    start_entity = start_entity.values
    start_entity = start_entity[:,0]

    with open(data_path, 'rb') as f:
        objects = ijson.items(f, 'item')
        question_split_prompt_generate_api(objects,start_entity)