model_type=$1
model_name_or_path=$2
dataset_name=$3

for entry in "data/"$dataset_name"/train_relation_wise/"*.json
do
    filename=$(basename $entry)
    rel_id="${filename%.*}"

    if [ "$rel_id" != "all" ]; then
        bash scripts/factual_knowledge_probing/prompt_tuning/single_relation/prompt_tuning.sh $model_type $model_name_or_path $dataset_name $rel_id
    fi
done