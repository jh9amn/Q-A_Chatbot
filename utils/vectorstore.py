from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.embeddings import get_embeddings

def create_vectorstore(docs):
    if not docs:
        raise ValueError("No documents to index. Content loader returned empty data.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    
    split_docs = text_splitter.split_documents(docs)
    split_docs = [s for s in split_docs if s.page_content.strip()]
    if not split_docs:
        raise ValueError("Text splitter produced no chunks.")


    
    vectordb = Chroma.from_documents(
        documents = split_docs,
        embedding = get_embeddings(),
        collection_name = "my_vectorstore",
        persist_directory = "db"
    )
    
    return vectordb