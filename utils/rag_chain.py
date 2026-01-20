import pydantic
import langchain

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from utils.llm import get_llm

def create_rag_chain(vectordb):
    llm = get_llm()
    
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages = True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectordb.as_retriever(),
        memory = memory,
        verbose = True
    
    )
    
    return chain