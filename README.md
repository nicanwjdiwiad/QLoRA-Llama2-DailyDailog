# QLoRA-Llama2-DailyDailog
# QLoRA微调项目

## 项目背景
本项目旨在初次尝试QLoRA微调，并构建一个根据双方对话识别当前情感（emotion）、行为（act）、话题（topic）的模型，作为后续生成模型的输入。与端到端模型对数据集质量的高要求不同，采用二阶段拆分的方法能使得模型产生的劝解语句更加适用于真实情况。学习参考：https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html


## 设备与环境
- **设备**: L4-GPU  
- **显存**: 22GB(21.2GB used)  
- **框架**: PyTorch

## 超参数配置

### 基本超参数
- **Batch Size**: 72=12*6
- **Gradient Accumulation Steps**: 6
- **Learning Rate**:  5e-4
  - 基于https://rentry.org/llm-training#low-rank-adaptation-lora_1调整后的学习率（adjusted_lr）由以下公式计算：  
  ```math
    \text{adjusted\_lr} = \text{base\_lr} \times \sqrt{\frac{\text{supervised\_tokens\_in\_batch} \times \text{total\_supervised\_tokens}}{\left(\frac{\text{num\_steps}}{\text{num\_epochs}}\right) \times \text{pretrained\_bsz}}}
  ```
Meta官方并未给出预训练数据集(pretrained\_bsz)的大小，但是根据@mlabonne佬先前脚本中设定的参数大致可计算我需要的learning_rate
### QLoRA参数
- **r (低秩矩阵的秩)**: 64
- **α (量化精度)**: 16
- **目标模型**: attention(q,v)
