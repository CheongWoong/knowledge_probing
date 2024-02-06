model_type=$1
model_name_or_path=$2
dataset_name=$3
model_name=$(basename $model_name_or_path)
training_type="prompt_tuning"

nohup bash scripts/factual_knowledge_probing/prompt_tuning/all_relations/prompt_tuning_ds_zero2.sh $model_type $model_name_or_path $dataset_name > "results/logs/"$training_type"_log."$model_name &