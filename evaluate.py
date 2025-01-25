from transformers import pipeline

# Initialize the pipeline for text generation
def generate_text(prompt, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result

# Example usage
prompt = "What is a large language model?"
result = generate_text(prompt, model, tokenizer)
print(result[0]['generated_text'])
