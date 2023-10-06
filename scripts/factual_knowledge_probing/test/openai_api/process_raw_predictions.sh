filepath=$1
dataset_type=$2

python -m src.factual_knowledge_probing.openai_api.process_raw_predictions --file_path $filepath --dataset_type $dataset_type