model_name_or_path=$1
dataset_type=$2
dataset_name=$3

python -m src.factual_knowledge_probing.aggregate_predictions_for_prompt_tuning --file_path $model_name_or_path --dataset_type $dataset_type --dataset_name $dataset_name