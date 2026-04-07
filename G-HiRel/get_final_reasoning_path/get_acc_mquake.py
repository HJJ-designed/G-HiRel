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
        for case_id, object in enumerate(objects):

            questions = object.get("questions")

            correct_ans = object.get("new_answer")
            correct_ans_alias = object.get("new_answer_alias")

            error_count = 0

            for question_i, quesion in enumerate(questions):
                quesion_id = case_id * 3 + question_i

                ans_path = data_dict.get(str(quesion_id))
                if ans_path is None:
                    error_count += 1
                    continue

                if "->" in ans_path:
                    ans = ans_path.split("->")[-1].strip()
                elif "→" in ans_path:
                    ans = ans_path.split("→")[-1].strip()
                else:
                    ans = None

                if ans is None:
                    error_count += 1
                else:
                    if ans == correct_ans or ans in correct_ans_alias:
                        break
                    else:
                        error_count += 1

            if error_count == 3:
                error_case.append(case_id)

        print( len(error_case))
        print( 1.0 - len(error_case) / total_nums)
        print(error_case)

