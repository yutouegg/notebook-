## 1.什么是Lora  
### 1.1全量微调：
顾名思义，就是把大模型所有的参数都进行微调。但是随着大模型的发展，动辄大几十几百B，全量微调的压力也越发的大。  


### 1.2替代方案：
* Adapt Tuning：在训练中加入Adapt层，训练过程中固定其他参数只更新Adapt层的参数
*  P-tuning：他在每个问题之前添加一个可学习的前缀，就可以用来专门训练一个专业技能。如想微调大模型来专门回答历史问题，而不希望修改模型的所有参数。就可以使用P-tuning，在每个问题之前添加一个可学习的前缀，如 “历史专家：”  


### 1.3Lora微调：
LoRA 的核心思想是通过训练低秩矩阵捕捉任务特定的变化，而保持预训练模型的大部分参数不变。低秩矩阵表示信息的压缩，只需少量参数就能近似描述数据的主要特征。
假设你有一幅高分辨率的图片（预训练模型），但只想调整颜色（适应特定任务）。你不需要修改每个像素（全参数微调），而是可以应用一个简单的滤镜（低秩矩阵）来达到效果。这种滤镜可以用很少的参数描述，却能显著改变图片的整体色调。这就类似于 LoRA 的低秩适应方法。

## 2.Lora微调blibli大模型(PEFT)：
### 2.1模型下载： 
```commandline
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('IndexTeam/Index-1.9B-Chat', cache_dir='model_path', revision='master') #版本选master是因为一般master是稳定的版本
```
  
### 2.2指令集的构建  
训练集选择从网上找到一个关于甄嬛语气的训练集，内容如下：  

![训练集](/image/微调01.png)

其中，instruction 是用户指令，告知模型其需要完成的任务；input 是用户输入，是完成用户指令所必须的输入内容；output 是模型应该给出的输出。  

数据格式化代码如下：
```commandline
def process_func(example):
    MAX_LENGTH = 384    # 分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<unk>system现在你要扮演皇帝身边的女人--甄嬛reserved_0user{example['instruction'] + example['input']}reserved_1assistant", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
```
这个训练集的构建也有点像P—tuning，在训练中加了前缀“现在你要扮演皇帝身边的女人--甄嬛”
process_func函数返回一个字典包含 input_ids、attention_mask 和 labels三个部分：  
* input_ids：将输入的文本编码传给模型
* attention_mask：表示模型需要关注哪些地方，一般填充的不太需要关注
* 回答的文本编码
  

### 2.3加载模型和Tokenizer
```commandline
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Tsumugii24/Index-1.9B-Chat/', use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/Tsumugii24/Index-1.9B-Chat/', device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
```  

### 2.4定义LoraConfig:  
```commandline
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # 决定了低秩矩阵的维度
    lora_alpha=32, # 缩放因子
    lora_dropout=0.1 # Dropout 比例
)
```  

### 2.5定义Train的参数并训练
```commandline
args = TrainingArguments(
    output_dir="./output/Index-1.9B-Chat-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
   )
   
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

```
后等待训练就好了...  

### 2.6加载lora微调
微调完成后当然就得调用啦:
```commandline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = 'model_path'
lora_path = 'lora_path'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

prompt = "你是谁？"
messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```  

这就完成了基础的微调了，1.9B的模型训练集也不大，我在3090上训练不到半个小时就完成了一次微调。
后续我习惯是使用llama.cpp将模型量化传到ollama进行调用