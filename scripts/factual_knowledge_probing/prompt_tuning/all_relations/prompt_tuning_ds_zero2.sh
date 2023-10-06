model_name_or_path=$1
dataset_name="LAMA_TREx"

for entry in "data/"$dataset_name"/train_relation_wise/"*.json
do
    filename=$(basename $entry)
    rel_id="${filename%.*}"

    if [ "$rel_id" != "all" ]; then
        bash scripts/factual_knowledge_probing/prompt_tuning/single_relation/prompt_tuning_ds_zero2.sh $model_name_or_path $rel_id
    fi
done