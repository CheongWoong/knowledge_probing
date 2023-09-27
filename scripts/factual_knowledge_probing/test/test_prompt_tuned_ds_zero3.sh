model_name_or_path=$1
dataset_type=$2
model_name=$(basename $model_name_or_path)
dataset_name="LAMA_TREx"

nohup bash scripts/factual_knowledge_probing/test/prompt_tuning_all_relations/test_prompt_tuned_ds_zero3.sh $model_name_or_path $dataset_type > "results/logs/log."$model_name"_"$rel_id".test_"$dataset_name"_"$dataset_type &