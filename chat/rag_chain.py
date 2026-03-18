from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from chat.memory import get_memory

def build_rag_chain(retriever):
    memory = get_memory()

    return ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0),
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
