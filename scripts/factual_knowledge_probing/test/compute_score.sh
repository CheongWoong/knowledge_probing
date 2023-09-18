pred_file=$1
reference_file='data/LAMA_TREx/all.json'
python -m src.factual_knowledge_probing.compute_score --pred_file $pred_file --reference_file $reference_file