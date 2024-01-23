model_type=$1
model_name_or_path=$2
dataset_name=$3
dataset_type=$4

for entry in "data/"$dataset_name"/"$dataset_type"_relation_wise/"*.json
do
    filename=$(basename $entry)
    rel_id="${filename%.*}"

    echo $rel_id

    if [ "$rel_id" != "all" ]; then
        bash scripts/factual_knowledge_probing/test/prompt_tuning_single_relation/test_prompt_tuned.sh $model_type $model_name_or_path $dataset_name $dataset_type $rel_id
    fi
done