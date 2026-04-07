#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Site    :
# @Software: PyCharm

import logging
import os
import time
import warnings
import openai
import ijson
import torch

import sys

warnings.filterwarnings("ignore")

cuda_num = 0
torch.cuda.set_device(cuda_num)

def get_target_relationships(MQuAKE_data, init_prompt, max_retries = 5, delay_seconds = 3):
    for question_id, question in enumerate(MQuAKE_data):
        prompt = init_prompt + f"\n\nQuestion {question_id} sentence: {question}" +\
                 f"\nQuestion {question_id}: Based on the relationships defined in **Wikidata**, what is the target relationship of <Question {question_id} sentence> ?"+\
                 f"\nAnswer {question_id}:"

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

        answer_1 = answer.split("Now is start, please answer the question as the examples and do not output any extra information.\n\n")[-1]
        answer_2 = answer_1.split("\n\n")[0]
        print(f"Answer {question_id}: {answer_2}")
        print("- "*80)


if __name__ == "__main__":
    openai.api_key = "APIKEY"

    init_prompt_path = r"prompts/prompt.txt"
    MQuAKE_path = r"../../data/MQUAKE/MQuAKE-CF-3k.json"
    MQuAKE_T_path = r"../../data//MQUAKE-T/MQuAKE-T.json"

    print("init_prompt_path:",init_prompt_path)
    print("MQuAKE_path:",MQuAKE_path)
    print("MQuAKE_T_path:",init_prompt_path)

    MQuAKE_data = []
    with open(MQuAKE_path, 'rb') as f:
        mquake_objects = ijson.items(f, 'item')

        for mquake_object in mquake_objects:
            MQuAKE_data = MQuAKE_data + mquake_object.get("questions")

    with open(MQuAKE_T_path, 'rb') as f:
        mquake_objects = ijson.items(f, 'item')

        for mquake_object in mquake_objects:
            MQuAKE_data = MQuAKE_data + mquake_object.get("questions")


    with open(init_prompt_path, 'r', encoding='utf-8') as file:
        init_prompt = file.read()

    get_target_relationships(MQuAKE_data, init_prompt)
