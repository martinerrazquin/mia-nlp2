import re
from dotenv import load_dotenv
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

PROMPT = """
Eres un asistente que responde preguntas sobre un curriculum vitae (CV).
Tu tono es profesional y tus respuestas son concisas, únicamente basadas en información relevada.
No ofreces hacer cosas extra al final de tus respuestas, ya que si el usuario lo quisiera lo hubiera preguntado.
En caso de que la información relevada no permita responder la pregunta del usuario, simplemente informas:
"No encontré información suficiente para responder su pregunta".

Información relevada:
{context}

Pregunta del usuario:
{question}
"""

class ChatbotConfig(BaseModel):
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_index_namespace: str
    groq_api_key: str
    llm_model: str
    llm_temperature: float
    search_top_k: int

class RAGResponse(BaseModel):
    answer: str
    reasoning: str | None = None


class Chatbot:
    def __init__(self, config: ChatbotConfig):
        """
        Initializes the chatbot with the provided configuration.

        This method sets up the Pinecone index, the Groq language model (LLM), and the chain of operations
        for retrieving relevant context from the Pinecone index and passing the question through the 
        LLM to generate a response. It also defines how to query the Pinecone index for context based 
        on the user's question.

        Args:
            config (ChatbotConfig): The configuration object containing necessary API keys, model settings, 
                                     and Pinecone index details.
        """
        # init index
        pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = pc.Index(config.pinecone_index_name)
        
        # init llm
        self.llm = ChatGroq(
            api_key=config.groq_api_key, # type: ignore
            model=config.llm_model,
            temperature=config.llm_temperature,
        )

        # init chain

        # util function for retrieval, VectorStore doesn't work nicely
        def retrieve(user_question):
            results = self.index.search(
                namespace=config.pinecone_index_namespace,
                query={
                    "inputs": {"text": user_question}, 
                    "top_k": config.search_top_k
                }, # type: ignore
                fields=["text"]
            )
            # parse -> join as str
            return "\n\n".join(
                result['fields']['text']
                for result in results['result']['hits']
            )

        self.chain = (
            {
                "context": RunnableLambda(retrieve),
                "question": RunnablePassthrough()
            }
            | ChatPromptTemplate.from_template(PROMPT)
            | self.llm
            | StrOutputParser()
        )

    def answer(self, user_question: str) -> RAGResponse:
        """
        Generates an answer to the user's question by processing the input through the chatbot's chain.

        This method invokes the chain of operations defined in the constructor to retrieve relevant context 
        from the Pinecone index, pass the question and context through the language model, and return a final 
        answer. Additionally, it extracts any reasoning behind the answer, if available.

        Args:
            user_question (str): The question asked by the user.

        Returns:
            RAGResponse: The response object containing the generated answer and any associated reasoning.
        """
        llm_resp = self.chain.invoke(user_question)

        # parse thinking tags using regex
        match = re.search(r"<think>(.*?)</think>", llm_resp, re.DOTALL)

        reasoning = match.group(1).strip() if match else None

        answer = re.sub(
            r"<think>.*?</think>",
            "",
            llm_resp,
            flags=re.DOTALL
        ).strip()
        
        return RAGResponse(answer=answer, reasoning=reasoning)
