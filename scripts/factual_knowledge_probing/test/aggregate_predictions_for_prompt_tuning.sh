file_path=$1
dataset_type=$2

python -m src.factual_knowledge_probing.aggregate_predictions_for_prompt_tuning --file_path $file_path --dataset_type $dataset_type