model_type=$1
model_name_or_path=$2
dataset_name=$3
rel_id=$4
model_name=$(basename $model_name_or_path)
training_type="prompt_tuning"
out_dir=$model_name"_"$dataset_name"_"$training_type

python -m "src.factual_knowledge_probing.run_"$model_type \
    --model_name_or_path $model_name_or_path \
    --do_train True \
    --do_eval True \
    --train_file "./data/"$dataset_name"/train_relation_wise/"$rel_id".json" \
    --validation_file "./data/"$dataset_name"/test_relation_wise/"$rel_id".json" \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --fp16 True \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --block_size 128 \
    --save_strategy no \
    --seed 0 \
    --report_to tensorboard \
    --output_dir "results/"$out_dir"/"$rel_id \
    --prompt_tuning True \
    > "results/logs/log."$out_dir"_"$rel_id