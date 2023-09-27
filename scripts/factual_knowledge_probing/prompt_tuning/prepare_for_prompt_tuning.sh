dataset_name="LAMA_TREx"
data_path="data/"$dataset_name

python -m src.factual_knowledge_probing.prepare_for_prompt_tuning --data_path $data_path