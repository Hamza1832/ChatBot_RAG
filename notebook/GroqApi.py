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

"""## Test Groq """
#https://console.groq.com/keys
#https://console.groq.com/playground
GROQ_API_KEY ="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"
groq_llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0,  api_key=GROQ_API_KEY)
system_message ="""
Classify the sentiment of the review presented in the input as 'positive' or 'negative'
The review will be delimited by triple backticks ``` in the input.
Answer only 'positive' or 'negative'
Do not explain your answer.
"""
user_message_template ="```{review}```"
user_message ="I think that your services are very fine"
zero_shot_prompt = [
    {"role":"system","content":system_message},
    {"role":"user", "content":user_message_template.format(review=user_message)},
]
response = groq_llm.invoke(zero_shot_prompt)
print(response.content.replace("</s>",""))
