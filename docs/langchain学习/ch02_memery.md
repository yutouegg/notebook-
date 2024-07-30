## 1.ConversationBufferMemory:  
### 导入相关模块：
```commandline
    from langchain.chains.conversation.memory import ConversationBufferMemory
    from langchain_openai import OpenAI
    from langchain.chains import ConversationChain
```
### 进行问答：
```commandline
    llm = OpenAI( 
             temperature=0, 
             max_tokens = 256)
     memory = ConversationBufferMemory()
     conversation = ConversationChain(
                  llm=llm, 
                  verbose=True, 
                   memory=memory)
     conversation.predict(input="你好")
```
---
*ConversationBufferMemory会将用户与ai的问答和下一次的propmt一起传给大模型进行问答，这样的缺点就是很快会超出token*  

---

## 2.ConversationSummaryMemory  
```commandline
    # memery替换成ConversationSummaryMemory
    memory = ConversationSummaryMemory()
    conversation = ConversationChain(
                  llm=llm, 
                  verbose=True, 
                   memory=memory)
     conversation.predict(input="你好")
```

---
*顾名思义，ConversationSummaryMemory会将用户与ai的对话进行总结进行记忆再传给大模型

---

* 提示：在langchain0.1.17版本以后使用langchain.chains会报警告langchain.chains已经不再使用