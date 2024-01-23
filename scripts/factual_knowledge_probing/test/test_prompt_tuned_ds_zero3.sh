model_type=$1
model_name_or_path=$2
dataset_name=$3
dataset_type=$4
model_name=$(basename $model_name_or_path)

nohup bash scripts/factual_knowledge_probing/test/prompt_tuning_all_relations/test_prompt_tuned_ds_zero3.sh $model_type $model_name_or_path $dataset_name $dataset_type > "results/logs/log."$model_name".test_"$dataset_name"_"$dataset_type &