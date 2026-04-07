#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Site    :
# @Software: PyCharm

import ijson
import pandas as pd
import openai
import time

def get_answer(prompt, max_retries = 5, delay_seconds = 3):

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

    if response['choices'][0]['message'].get("content") is not None:
        gpt_answer = response['choices'][0]['message']['content']
    else:
        gpt_answer = None

    return gpt_answer


def get_entity_prompt_generate(objects):
    init_Prompt = "For the following questions, we extract the initial entities involved in the question through the entity set " \
                  "based on **Wikidata**. " \
                  "\n3011.Question: What is the nationality of the author of \"Harry\"?" \
                  "\n3011.The Question is Start of: Harry" \
                  "\n4025.Question: What continent is the country of origin of Michael Jordan\'s sport located in?" \
                  "\n4025.The Question is Start of: Michael Jordan" \
                  "\n4086.Question: Which city in the country of origin is associated with the West Coast Hip Hop genre?" \
                  "\n4086.The Question is Start of: West Coast Hip Hop"
    prompt = init_Prompt

    for i, obj in enumerate(objects):
        question_prompt = f"\n{i + 1}.Question: " + obj.get("questions")[0]
        startEntity_prompt = "\nn{i + 1}.The Question is Start of: "
        prompt = prompt + question_prompt + startEntity_prompt

        start_entity = get_answer(prompt)
        print(f"{i}. start_entity: {start_entity}")

        prompt = init_Prompt
        print("- " * 40)

if __name__=="__main__":
    openai.api_key = "APIKEY"

    mquake_path = r"../../data/MQUAKE/MQuAKE-CF-3k.json"
    # mquake_path = r"../../data/MQUAKE-T/MQuAKE-T.json"

    with open(mquake_path, 'rb') as f:
        objects = ijson.items(f, 'item')

        get_entity_prompt_generate(objects)