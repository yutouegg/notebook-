## 通过在线和本地调用两种方式分别来实现

### 1.通过在线的方式调用google/flan-t5-xl进行问答（结合提示词）
```commandline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
#定义链
llm_chain = LLMChain(prompt=prompt, 
                     llm=HuggingFaceHub(repo_id="google/flan-t5-xl", 
                                        model_kwargs={"temperature":0, 
                                                      "max_length":64}))
question = "小米集团的CEO是谁?"

print(llm_chain.invoke(question))
```  
---

_通过在线调用的方式可能会出现网络和token等各种莫名其妙的错误，所以一般我不怎么使用huggingface在线调用的方式_

### 2.使用lacal model
```commandline
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)#使用8bit量化

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

question = "谁是小米集团的CEO?"

print(llm_chain.invoke(question))
```

当然使用Hugingfacehub也一样得有Huggingface的key，我通常喜欢用os将key保存在环境中