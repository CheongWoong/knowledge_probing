model_name_or_path=$1
model_name=$(basename $model_name_or_path)
training_type="prompt_tuning"

nohup bash scripts/factual_knowledge_probing/prompt_tuning/all_relations/prompt_tuning.sh > "results/logs/"$training_type"_log."$model_name &