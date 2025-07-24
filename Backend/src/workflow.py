from langchain_groq import ChatGroq
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils import (save_uploaded_file,getRetreiver,query_split)
import os
from dotenv import load_dotenv
import time
load_dotenv()

llm=ChatGroq(model="gemma2-9b-it",api_key=os.getenv("GROQ_API_KEY"))
def get_Response(query,document):
    start = time.process_time()
    retreiver=getRetreiver(save_uploaded_file(document))
    que=query_split(query)
    docs=[]
    for q in que:
        docs.extend(retreiver.get_relevant_documents(q))
    prompt=ChatPromptTemplate([
        ("system", """You are an official assistant of Bajaj Finance. You have been asked a question based on a document 
        or policy the user has uploaded. I have provided you relevant documents based on the user query.
        Please check whether the query values like age, sex, injury type, location, and policy duration lie within the policy or not.
        Respond in 2–3 lines professionally — like: 'Yes, this policy covers all mentioned conditions' or something similar.
        
        Context:
        {context}
        """),
        ("user", "query: {query}")
    ])
    if(len(que)==4):
        query_formation=f"""
    Age and Sex={que[0]},
    Type of Injury={que[1]},
    Location of residency={que[2]},
    Time_limit of policy={que[3]}
    """
    else:
        query_formation=query
    doc_chain=create_stuff_documents_chain(llm,prompt)
    raw=doc_chain.invoke({"query":query_formation,"Context":docs})
    response = StrOutputParser().invoke(raw)

    end=time.process_time()
    print("Time processed:",end-start)
    return response

