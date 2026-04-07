#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Software: PyCharm

import csv
import ijson

if __name__== "__main__":

    file_path = "./results/MQuAKE-CF-3k/edit_1_topK_3.csv" # result file path
    mquake_data_path = r"../data/MQUAKE/MQuAKE-CF-3k.json"# dataset path

    total_nums = 3000
    if "-T" in mquake_data_path:
        total_nums = 1868

    data_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            line = row[0]
            if "." in line:
                index_str, content = line.split(".", 1)
                index = index_str.strip()
                if data_dict.get(index) is None:
                    data_dict[index] = content.strip()

    error_case = []
    with open(mquake_data_path, "r", encoding="utf-8") as f:
        objects = ijson.items(f,'item')
        hop_correct_count = 0

        for case_id, obj in enumerate(objects):
            questions = obj.get("questions", [])
            correct_triples = obj.get("orig", {}).get("new_triples_labeled", [])
            gold_triples = [[item.strip() for item in triple] for triple in correct_triples]

            for question_i, _ in enumerate(questions):
                quesion_id = case_id * 3 + question_i
                pred_line = data_dict.get(str(quesion_id))

                if pred_line is None:
                    continue

                triple_strings = [s.strip() for s in pred_line.replace("→", "->").split(";") if s.strip()]
                pred_triples = []
                for triple_str in triple_strings:
                    parts = [p.strip() for p in triple_str.split("->")]
                    if len(parts) == 3:
                        pred_triples.append(parts)

                if pred_triples == gold_triples:
                    hop_correct_count += 1
                    break
        print(f"hop_Acc: {hop_correct_count / total_nums:.4f}")
