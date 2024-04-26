import os
from langchain_openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] ="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"

chat_history = []
vector_db = FAISS.load_local("dataset",HuggingFaceEmbeddings(),allow_dangerous_deserialization=True)
# qa = ConversationalRetrievalChain.from_llm(OpenAI(), vector_db.as_retriever())

retriever = vector_db.as_retriever()

qa = OpenAI(temperature=0.5)
conversational_qa = ConversationalRetrievalChain.from_llm(qa, retriever)

def get_response(query):
    prompt = f"Given the conversation history {chat_history} and the user query '{query}'. Try to provide accurate and helpful information related to legal matters."

    response = conversational_qa.invoke({"question": query, "chat_history": chat_history, "prompt": prompt})
    chat_history.append((query, response['answer']))
    print(query)
    print(response["answer"])
    return response["answer"]


   



