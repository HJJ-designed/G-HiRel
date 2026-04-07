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
    init_prompt_path = "./prompts/prompts.txt"

    with open(init_prompt_path, 'r', encoding='utf-8') as file:
        init_Prompt = file.read()

    prompt = init_Prompt + "\n"

    for i, obj in enumerate(objects):
        questions = obj.get("questions")
        for j in range(len(questions)):
            question_prompt = f"\n{i * 3 + 1 + j}.Text: " + questions[j]
            entity_prompt = f"\nStart Entity:{start_entities[i]}"
            prompt = prompt + question_prompt + entity_prompt

        response = get_answer(prompt)
        print(response)
        print("- " * 40)
        time.sleep(0.1)
        prompt = init_Prompt + "\n"



if __name__=="__main__":
    # Set the API key.
    openai.api_key = "APIKEY"

    data_path = r"../data/MQUAKE/MQuAKE-CF-3k.json"
    start_entity_path = r"../KG/start_entities/mquake_entity2qids.csv"

    # data_path = r"../data/MQUAKE-T/MQuAKE-T.json"
    # start_entity_path = r"../KG/start_entities/mquake_t_entity2qids.csv"
    start_entity = pd.read_csv(start_entity_path,header=None,encoding="UTF-8")
    start_entity = start_entity.values
    start_entity = start_entity[:,0]

    with open(data_path, 'rb') as f:
        objects = ijson.items(f, 'item')
        question_split_prompt_generate_api(objects,start_entity)