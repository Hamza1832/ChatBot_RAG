import openai
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import tiktoken
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import session_info
session_info.show()
"""## Test Ollama (Local Machine)"""
#install ollama : https://ollama.com/download
#https://ollama.com/library/mistral

llama_llm  = Ollama(model="mistral", temperature=0)
system_message ="""
Classify the sentiment of the review presented in the input as 'positive' or 'negative'
The review will be delimited by triple backticks ``` in the input.
Answer only 'positive' or 'negative'.
Do not explain your answer.
"""
user_message_template ="```{review}```"
user_message ="The look is bad"
few_shot_prompt = [
    {"role":"system","content":system_message},
    {"role":"user", "content":user_message_template.format(review=user_message)},
]
response = llama_llm.invoke(few_shot_prompt)
print(response)
