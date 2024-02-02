model_name_or_path=$1
dataset_name=$2
dataset_type=$3

python -m src.factual_knowledge_probing.aggregate_predictions_for_prompt_tuning --model_name_or_path $model_name_or_path --dataset_name $dataset_name --dataset_type $dataset_type