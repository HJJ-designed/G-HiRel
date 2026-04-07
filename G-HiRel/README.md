## Environment：

You can create the required environment for G-HiRel using the ``environment.yml'` file.

## Model Prepare:

The model referenced in our paper can be downloaded from the following link:

**GPT-J (6B)**: https://huggingface.co/EleutherAI/gpt-j-6b

**Vicuna (7B)**: https://huggingface.co/lmsys/vicuna-7b-v1.5

**DeepSeek (7B)**: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base

**Flan-T5-Large**: https://huggingface.co/google/flan-t5-large

**DeBERTa-v3-Large**: https://huggingface.co/microsoft/deberta-v3-large

**all-RoBERTa-Large-v1**: https://huggingface.co/sentence-transformers/all-roberta-large-v1

**msmarco-MiniLM-L6-v3**: https://huggingface.co/sentence-transformers/msmarco-MiniLM-L6-v3

In addition, **GPT-4o-mini** and **DeepSeek-R1** are accessed via API.

## To run G-HiRel:

First, use the code in `G-HiRel/KG` to extract relation information from the Wikidata JSON file, build the foundational knowledge graph (KG), and identify the starting entity for each question in the MQuAKE dataset.  

Next, run the code in `G-HiRel/QuestionReformulation` to reformulate the test questions.  

Then, use the scripts in `G-HiRel/get_subGraph` to construct the reasoning knowledge graph.  

In addition, run the code in `G-HiRel/get_target_relation' to prepare the information needed for filtering the final relation paths.

Finally, run the code in `get_subGraph/get_final_relation_path` and `G-HiRel/get_final_reasoning_path` to obtain the relation path and answer path for each question.


