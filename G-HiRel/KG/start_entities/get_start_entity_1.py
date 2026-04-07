#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Site    :
# @Software: PyCharm
import pandas as pd
import re

file_path = "The_txt_file_collect_the_the_output_of_get_start_entity"
csv_output_path = "./start_entity_name_mquake_t.csv"

with open(file_path, "r", encoding="utf-8") as file:
    data = file.readlines()

entities = []
for row in data:
    if("- - - ") in row:
        continue
    else:
        entities.append(row.split("start_entity:")[-1].replace("\n","").strip())

df = pd.DataFrame(entities, columns=["entity_name"])
df.to_csv(csv_output_path, index=False, encoding="utf-8")
print(f"Entities has been saved to {csv_output_path}")