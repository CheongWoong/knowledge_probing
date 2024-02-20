filepath=$1
dataset_name=$2
dataset_type=$3

python -m src.factual_knowledge_probing.openai_api.process_raw_predictions --file_path $filepath --dataset_name $dataset_name --dataset_type $dataset_type