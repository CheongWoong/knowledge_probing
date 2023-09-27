model_name_or_path=$1
dataset_type=$2
dataset_name="LAMA_TREx"

for entry in "data/"$dataset_name"/"$dataset_type"_relation_wise/"*.json
do
    filename=$(basename $entry)
    rel_id="${filename%.*}"

    echo $rel_id

    if [ "$rel_id" != "all" ]; then
        bash scripts/factual_knowledge_probing/test/prompt_tuning_single_relation/test_prompt_tuned.sh $model_name_or_path $dataset_type $rel_id
    fi
done