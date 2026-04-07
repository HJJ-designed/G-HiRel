#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Site    :
# @Software: PyCharm

import csv
import os
import time
import ijson
import pandas as pd


batch_size = 1_0000
batches = []
wiki_data_path = r""  # Wikidata file path
relation_path = r"" # relation extraction result path


triples_write_path = r"../data/wikidata/triples.csv"  # Storage path of the extracted triples.
items_write_path = r"../data/wikidata/items_to_label_and_alias.csv"  # Mapping path between Qids and names.
property_write_path = r"../data/wikidata/properties_to_label_and_alias.csv"  # Mapping path between Pids and names.

df = pd.read_csv(relation_path)
property_list = df["property"].tolist()

P_list = list(set(property_list))
print(len(P_list))

triples_head = ['subject', 'property', 'object']
items_head = ["Qid", "name", "describe"]
property_head = ["Pid", "name", "describe"]


def write_to_csv(list_elements, output_path, head):
    file_exists = os.path.isfile(output_path)
    with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        if not file_exists:
            csv_writer.writerow(head)

        for triple in list_elements:
            csv_writer.writerow(triple)


def get_info_from_sample(sample):
    id_cur = sample.get("id")

    id_name = sample.get("labels")
    id_name_final = "EmptyName"  # Use a placeholder when the name is unavailable.
    if id_name is not None:
        id_name = id_name.get("en")
        if id_name is not None:
            id_name_final = id_name.get("value")

    id_describe = sample.get("descriptions")
    id_describe_final = "EmptyDescribe"  # Use a placeholder when the name is unavailable.
    if id_describe is not None:
        id_describe = id_describe.get("en")
        if id_describe is not None:
            id_describe_final = id_describe.get("value")

    return (id_cur, id_name_final, id_describe_final)


def process_batch_data():
    triples = []
    Qids_name_describe = []
    Properities_name_describe = []

    for idx, sample in enumerate(batches, 1):
        s_id = sample.get('id')
        claims = sample.get('claims')
        if claims is not None:
            for r_id, claim_list in claims.items():
                if r_id in P_list:
                    for claim in claim_list:
                        mainsnak = claim.get("mainsnak")
                        if mainsnak:
                            datavalue = mainsnak.get("datavalue")
                            if datavalue:
                                o_value = datavalue.get("value")
                                if isinstance(o_value, dict) and o_value.get("entity-type") == "item":
                                    o_id = f"Q{o_value.get('numeric-id')}"
                                    if s_id.startswith("Q") and o_id.startswith("Q"):
                                        triples.append((s_id, r_id, o_id))
                                elif isinstance(o_value, str):
                                    if s_id.startswith("Q") and o_value.startswith("Q"):
                                        triples.append((s_id, r_id, o_value))

        if sample.get("type") == "item":
            triple_cur = get_info_from_sample(sample)
            Qids_name_describe.append(triple_cur)

        elif sample.get("type") == "property":
            triple_cur = get_info_from_sample(sample)
            Properities_name_describe.append(triple_cur)

    write_to_csv(triples, triples_write_path, triples_head)
    write_to_csv(Qids_name_describe, items_write_path, items_head)
    write_to_csv(Properities_name_describe, property_write_path, property_head)


if __name__ == "__main__":
    start_time = time.time()

    with open(wiki_data_path, 'rb') as f:
        objects = ijson.items(f, 'item')
        for i, obj in enumerate(objects):
            batches.append(obj)
            if (i + 1) % batch_size == 0:
                process_batch_data()
                batches = []
                print(f"{((i+1) / 1_0000):.1f} w has been preprcessed，time cost：{(time.time() - start_time):.2f} s")
