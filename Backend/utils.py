from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers
import tempfile
import os

def save_uploaded_file(file):

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.name)

    # Save the file content
    with open(file_path, "wb") as f:
        f.write(file.read())

    return file_path


def getRetreiver(path):
    loader=PyPDFLoader(path)
    doc=loader.load()
    doc=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(doc)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store=FAISS.from_documents(documents=doc, embedding=embeddings)
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 7})

def query_split(query):
    qu=query.split(",")
    return qu