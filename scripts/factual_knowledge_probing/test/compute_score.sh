pred_file=$1
dataset_name=$2
reference_file="data/"$dataset_name"/all.json"

python -m src.factual_knowledge_probing.compute_score --pred_file $pred_file --reference_file $reference_file