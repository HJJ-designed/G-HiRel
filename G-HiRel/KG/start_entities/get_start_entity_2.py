import pandas as pd


# Read the Qid2Name file.
qid_Name_path = r"../../data/wikidata/items_to_label_and_alias.csv"

# Read the entity_name file and the corresponding output file.
entity_name_path = r"./start_entity_name_mquake_t.csv"
output_csv_path = "./mquake_t_entity2qids.csv"

# entity_name_path = r"./start_entity_name_mquake.csv"
# output_csv_path = "./mquake_entity2qids.csv"


qid_Name = pd.read_csv(qid_Name_path, encoding="UTF-8")
entity_name = pd.read_csv(entity_name_path, encoding="utf-8")

qid_Name['name'] = qid_Name['name'].str.strip().str.lower()
entity_name['entity_name'] = entity_name['entity_name'].str.strip().str.lower()

entity_name['entity_name'] = (entity_name['entity_name']
                              .str.replace('"', "", regex=False))

matched_df = qid_Name[qid_Name['name'].isin(entity_name['entity_name'])]

qid_mapping = matched_df.groupby('name')['Qid'].apply(lambda x: ';'.join(map(str, x))).reset_index()

final_result = entity_name.merge(qid_mapping, left_on='entity_name', right_on='name', how='left').drop(columns=['name'])

final_result['Qid'] = final_result['Qid'].fillna("None")

final_result.loc[
    final_result['entity_name'].isna() |
    (final_result['entity_name'] == "") |
    (final_result['entity_name'] == "none"),
    ['entity_name', 'Qid']
] = "None"

final_result.to_csv(output_csv_path, index=False, header=False, encoding="utf-8")

print(f"Processing completed. Results have been saved to {output_csv_path}")



