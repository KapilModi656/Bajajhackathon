from langchain_groq import ChatGroq
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from Backend.utils import save_uploaded_file, getRetreiver, query_split
import os
from dotenv import load_dotenv
import time
import sys
from Backend.exception import CustomException
import Backend.custom_logging as custom_logging
from langchain_core.documents import Document

logger = custom_logging.logging.getLogger()
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")

llm = ChatGroq(model="gemma2-9b-it") 

async def get_Response(query, document):
    try:
        logger.info("Starting the response generation process")
        start = time.process_time()

        file_path = await save_uploaded_file(document)
        retriever = getRetreiver(file_path)

        logger.info("Retrieving relevant documents for the query")
        que = query_split(query)
        
        
        

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an official assistant of Bajaj Finance. 
             Based on the provided policy documents, answer clearly and 
             concisely whether the query values like age, sex, injury type, location, and 
             policy duration are covered.

- If covered, respond with confirmation and relevant details in 2-3 lines.
- If not covered or unclear, respond: "The policy does not cover this."
- Provide examples of queries and answers for clarity and these are samples 
             like how you would have to behave dont rote this answer check from context if there then only write it is covered.
Example 1:
Query: Age=46F, Injury=Brain surgery, Location=Mumbai, Policy Duration=3-month
Answer: Yes, Brain surgery is covered under the policy for residents of Mumbai with a 
             3-month duration.

Example 2:
Query: Age=30F, Injury=heart transplant, Location=Mumbai, Policy Duration=1-year
Answer: The policy does not cover heart transplant procedures.

Context:
{context}
"""),
            ("user", "query: {query}")
        ])

      
        query_formation = (
            f"Age and Sex={que[0]}, Type of Injury={que[1]}, Location of residency={que[2]}, Time_limit of policy={que[3]}"
            if len(que) == 4 else query
        )


        all_docs = []
        for q in que:
            result = retriever.invoke(q)
            if isinstance(result, list):
                all_docs.extend(result)
            else:
                all_docs.append(result)

       
        logger.info(f"Docs content types: {[type(d) for d in all_docs]}")
        logger.info(f"Docs preview: {all_docs[:3]}")

        context_docs = []
        for doc in all_docs:
            if hasattr(doc, "page_content") and getattr(doc, "page_content", None) is not None:
                context_docs.append(doc)
            else:
                logger.warning(f"Doc missing page_content: {doc} type: {type(doc)}")

        context = "\n\n".join(doc.page_content for doc in context_docs)
        print("Context:", context)

        logger.info("Creating document chain for processing the query")
        chain=prompt|llm

        raw_response = chain.invoke({"query": query_formation, "context": context})

        if isinstance(raw_response, str):
            response = raw_response
        elif isinstance(raw_response, dict):
            response = raw_response.get("output", str(raw_response))
        else:
            response = str(raw_response)

        end = time.process_time()
        logger.info("Time processed: %s", end - start)

        return response

    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        raise CustomException(e, sys) from e
