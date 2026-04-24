import os
from dotenv import load_dotenv
import streamlit as st

from src.chatbot import Chatbot, ChatbotConfig, RAGResponse


load_dotenv()
MODEL_NAME = "qwen/qwen3-32b" # fixed

def load_cfg() -> ChatbotConfig:
    cfg = ChatbotConfig(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
        pinecone_index_namespace=os.getenv("PINECONE_INDEX_NAMESPACE"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        llm_model=MODEL_NAME,
        llm_temperature=os.getenv("LLM_TEMPERATURE"),
        search_top_k=os.getenv("SEARCH_K_RESULTS"),
    )

    return cfg


@st.cache_resource
def load_bot() -> Chatbot:
    config = load_cfg()

    bot = Chatbot(config)

    return bot

bot = load_bot()


# UI
st.title("Consulta de CV")

question = st.text_input("Ingrese su pregunta:")

if st.button("Consultar") and question.strip():
    with st.spinner("Procesando..."):
        response = bot.answer(question)

    # main answer header
    st.subheader("Respuesta")

    # reasoning (hidden by default + expandable)
    if response.reasoning:
        with st.expander("Ver razonamiento"):
            st.write(response.reasoning)
    
    st.write(response.answer)

