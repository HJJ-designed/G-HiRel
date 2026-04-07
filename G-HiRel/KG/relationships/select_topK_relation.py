#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Software: PyCharm

import re
from collections import Counter
import pandas as pd

def parse_txt_to_dict(file_path):
    data_dict = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if " (P" in line:
                key, value = line.split(" (P")
                data_dict[key] = "P" + value.rstrip(")")
    return data_dict


def split_dict_keys(data_dict, num_groups=1):
    keys = list(data_dict.keys())
    group_size = len(keys) // num_groups

    groups = [keys[i * group_size:(i + 1) * group_size] for i in range(num_groups - 1)]
    groups.append(keys[(num_groups - 1) * group_size:])

    return groups


def extract_answers(file_path):
    answers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        matches = re.findall(r'Answer \d+: \{(.*?)\}', content, re.DOTALL)

        for match in matches:
            cleaned_match = match.replace('\n', '').strip()
            answers.append(cleaned_match)

    return answers


def get_top_n_relationships(total_relationships_extract, N):
    counter = Counter(total_relationships_extract)
    top_n = counter.most_common(N)
    return top_n


if __name__=="__main__":
    file_path = "../../data/wikidata/properties_to_label_and_alias.csv"
    relation_file_path = r"pid2Name/runInfo_gpt_4o_mini.txt"
    file_output_path = r"pid2Name/pid2Name.csv"

    data = pd.read_csv(file_path, encoding="utf-8")
    data_dict = data.set_index('name')['Pid'].to_dict()
    groups = split_dict_keys(data_dict, num_groups= 1)


    total_relationships_extract = []
    answer_list = extract_answers(relation_file_path)
    for string_item in answer_list:
        string_item_temp1 = string_item.split("\", \"")
        for item in string_item_temp1:
            item = item.replace("\"", "")
            if item in groups[0]:
                total_relationships_extract.append(item)

    top_n_relationships = get_top_n_relationships(total_relationships_extract, 50)
    output_df = pd.DataFrame(columns=["property", "propertyLabel"])

    data_list = []
    for item in top_n_relationships:
        if "," in item[0]:
            data_list.append({"property": data_dict.get(item[0]), "propertyLabel": item[0]})
        else:
            data_list.append({"property": data_dict.get(item[0]), "propertyLabel": item[0]})

    output_df = pd.DataFrame(data_list)
    output_df.to_csv(file_output_path, index=False, encoding="utf-8")
    print(f"CSV file has been saved to {file_output_path}")


