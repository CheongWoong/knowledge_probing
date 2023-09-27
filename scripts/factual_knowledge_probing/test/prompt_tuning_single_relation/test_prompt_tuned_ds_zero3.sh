model_name_or_path=$1
dataset_type=$2
rel_id=$3
test_rel_id=$rel_id
model_name=$(basename $model_name_or_path)
dataset_name="LAMA_TREx"
out_dir=$model_name
ds_zero_stage=3

deepspeed src/factual_knowledge_probing/run_factual_knowledge_probing.py \
    --deepspeed "scripts/factual_knowledge_probing/ds_config_zero"$ds_zero_stage".json" \
    --model_name_or_path $model_name_or_path"/"$rel_id \
    --do_train False \
    --do_eval True \
    --validation_file "./data/"$dataset_name"/"$dataset_type"_relation_wise/"$test_rel_id".json" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --fp16 True \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --block_size 128 \
    --save_strategy no \
    --seed 0 \
    --report_to tensorboard \
    --output_dir "results/"$out_dir"/"$rel_id \
    --prompt_tuning True \
    > "results/logs/log."$out_dir"_"$rel_id".test_"$dataset_name"_"$dataset_type"_"$test_rel_id