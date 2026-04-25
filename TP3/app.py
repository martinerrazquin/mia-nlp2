import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from src.agent import Agent, AgentConfig


load_dotenv()

MODEL_NAME = "qwen/qwen3-32b" # fixed
docs_path = Path("./docs")


def format_name(name_surname: tuple[str,str]) -> str:
    name, surname = name_surname
    return f"{name.capitalize()} {surname.capitalize()}"

@st.cache_resource
def get_users() -> list[tuple[str,str]]:
    return [
        tuple(p.stem.lower().split("_"))
        for p in sorted(docs_path.iterdir())
    ]

def load_cfg(all_users: list[tuple[str,str]]) -> AgentConfig:
    cfg = AgentConfig(
        name_surname_list=all_users,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_namespace=os.getenv("PINECONE_INDEX_NAMESPACE"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        llm_model=MODEL_NAME,
        llm_temperature=os.getenv("LLM_TEMPERATURE"),
        search_top_k=os.getenv("SEARCH_K_RESULTS"),
        
    )

    return cfg


@st.cache_resource
def load_agent(all_users: list[tuple[str,str]]) -> Agent:
    config = load_cfg(all_users)

    agent = Agent(config)

    return agent

all_users = get_users()
agent = load_agent(all_users)


# UI
st.title("Consulta de Múltiples CV")

selected_user = st.selectbox("Elegir usuario:", all_users, format_func=format_name)

st.write(f"Usuario seleccionado: **{format_name(selected_user)}**")

question = st.text_input("Ingrese su pregunta:")

if st.button("Consultar") and question.strip():
    with st.spinner("Procesando..."):
        result = agent.answer(question=question, user=selected_user)

    # main answer header
    st.subheader("Respuesta")

    st.write(f"La pregunta se interpreta sobre: **{format_name(result.detected_person)}**")

    # reasoning (hidden by default + expandable)
    if result.answer.reasoning:
        with st.expander("Ver razonamiento"):
            st.write(result.answer.reasoning)
    
    st.write(result.answer.answer)
