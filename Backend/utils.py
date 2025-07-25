from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os


async def save_uploaded_file(file):

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


def getRetreiver(path):
    if(path.endswith('.pdf')):
      
        loader = PyPDFLoader(file_path=path)
    elif(path.endswith('.txt') or path.endswith('.csv') or path.endswith('.docx') or path.endswith('docs')):
        loader = UnstructuredLoader(file_path=path)
    raw_docs = loader.load()  
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    split_docs = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)

    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
def query_split(query):
    qu=query.split(",")
    return qu