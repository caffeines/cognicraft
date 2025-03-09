import streamlit as st
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

st.title("Q&A Chatbot")
st.write("Welcome to the Q&A Chatbot! Ask me anything!")

prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI assistant that can answer any question."),
        ("user", "Question: {question}")
    ]
)

def get_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatOpenAI(model=llm, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    chain = prompt|llm|StrOutputParser()
    response = chain.invoke({"question": question})
    return response

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
llm = st.sidebar.selectbox("Language Model", ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 1, 2048, 100)

question = st.text_input("Ask a question")

if st.button("Get Answer") and question:
    response = get_response(question, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Click the button to get an answer")