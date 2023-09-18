from transformers import AutoModelForCausalLM


if __name__ == '__main__':
    model_paths = ['EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6b']
    output_paths = ['results/gpt_neo_125M', 'results/gpt_neo_1_3B', 'results/gpt_neo_2_7B', 'results/gpt_j_6B']
    model_paths = output_paths
    for model_path, output_path in zip(model_paths, output_paths):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.save_pretrained(output_path)