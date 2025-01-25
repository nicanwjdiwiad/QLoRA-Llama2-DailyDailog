def preprocess_function(examples):
    return {"text": [" ".join(dialog) for dialog in examples["dialog"]]}

# Tokenization step
def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
