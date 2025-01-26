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
new_model = "/content/drive/MyDrive/QLoRA-Llama2-dailydialog/llama-2-7b-QLoRA-emotion-1"  # 替换为你的微调后模型路径
after_model = AutoModelForCausalLM.from_pretrained(model, new_model) #PeftModel仅仅用于Huggingface的，这个可以用本地路径
after_model = model.merge_and_unload()
# 配置生成模型的pipeline
pipe = pipeline("text-generation", model=after_model, tokenizer=tokenizer, max_length=200)

# 给定对话
dialog = [
    'Say , Jim , how about going for a few beers after dinner ? ',
    ' You know that is tempting but is really not good for our fitness . ',
    ' What do you mean ? It will help us to relax . ',
    " Do you really think so ? I don't . It will just make us fat and act silly . Remember last time ? ",
    " I guess you are right.But what shall we do ? I don't feel like sitting at home . ",
    ' I suggest a walk over to the gym where we can play singsong and meet some of our friends . ',
    " That's a good idea . I hear Mary and Sally often go there to play pingpong.Perhaps we can make a foursome with them . ",
    ' Sounds great to me ! If they are willing , we could ask them to go dancing with us.That is excellent exercise and fun , too . ',
    " Good.Let ' s go now . ",
    ' All right . '
]

# 将对话拼接成一个单一的输入
input_text = " ".join(dialog)

# 生成模型输出（对话生成）
output = pipe(input_text)

# 输出模型生成的文本
print(f"Generated Output: {output[0]['generated_text']}")

# 对每个句子单独进行预测，得到标签输出（假设你的模型在微调时能够输出act和emotion标签）

# 示例：
for sentence in dialog:
    # 模型的生成结果（可能是对话输出）
    generated = pipe(sentence)
    
    # 这里你可以根据模型生成的输出进行后处理来得到act和emotion标签（如果你的微调过程中涉及到标签输出）
    print(f"Sentence: {sentence}")
    print(f"Generated: {generated[0]['generated_text']}")
    # 假设你微调时，模型也能够生成类似于 `act` 和 `emotion` 标签的信息
    # print(f"Predicted act: {generated[0]['act']}, Predicted emotion: {generated[0]['emotion']}")
