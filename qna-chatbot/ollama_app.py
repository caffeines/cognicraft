import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

st.title("Q&A Chatbot")
st.write("Welcome to the Q&A Chatbot! Ask me anything!")

prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI assistant that can answer any question."),
        ("user", "Question: {question}")
    ]
)

def get_response(question, llm, temperature):
    llm = ollama.Ollama(model=llm, temperature=temperature)
    chain = prompt|llm|StrOutputParser()
    response = chain.invoke({"question": question})
    return response

st.sidebar.title("Settings")
llm = st.sidebar.selectbox("Language Model", ["deepseek-r1:1.5b", "gemma2", "llama3"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

question = st.text_input("Ask a question")

if st.button("Get Answer") and question:
    response = get_response(question, llm, temperature)
    st.write(response)  
else:
    st.write("Click the button to get an answer")
