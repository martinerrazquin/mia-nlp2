import re
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from src.lookup_helper import PineconeLookup

load_dotenv()

PERSON_DETECTION_PROMPT = """
A continuación se presenta una lista de personas y una pregunta.
Tu tarea es determinar a cuál de las personas se refiere la pregunta, o si no especifica.
Tu respuesta debe ser estrictamente el número de la persona a la que hace referencia.
En caso de no se haga mención a nadie, tu respuesta debe ser estrictamente '0'.
No respondas más que el número.

EJEMPLO 1:
Lista de personas:
1) pedro ramirez
2) juana azurduy
3) sancho panza

Pregunta: ¿cómo se llama?
Razonamiento: en la pregunta no se especifica a quién se refiere, por lo que corresponde un 0
Respuesta esperada: 0

EJEMPLO 2:
Lista de personas:
1) pedro ramirez
2) juana azurduy
3) sancho panza

Pregunta: ¿de qué trabaja perdro?
Razonamiento: la pregunta menciona un 'perdro' que parece ser un error de tipeo, por lo tanto refiere a pedro ramirez, cuyo número es el 1
Respuesta esperada: 1

LISTA REAL:
{users}

PREGUNTA: {user_prompt}
"""


ANSWER_PROMPT = """
Eres un asistente que responde preguntas sobre un curriculum vitae (CV).
Tu tono es profesional y tus respuestas son concisas, únicamente basadas en información relevada.
No ofreces hacer cosas extra al final de tus respuestas, ya que si el usuario lo quisiera lo hubiera preguntado.
En caso de que la información relevada no permita responder la pregunta del usuario, simplemente informas:
"No encontré información suficiente para responder su pregunta".

Información relevada:
{retrieved_data}

Pregunta del usuario:
{user_prompt}
"""

class ThinkingResponse(BaseModel):
    reasoning: str | None = None
    answer: str

    @classmethod
    def from_llm_output(cls, text: str) -> "ThinkingResponse":
        pattern = r"<think>(.*?)</think>\s*(.*)"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
        else:
            reasoning = None
            answer = text.strip()

        return cls(reasoning=reasoning, answer=answer)

class AgentState(BaseModel):
    #messages: Annotated[list[AnyMessage], operator.add]
    user_prompt: str | None = None
    default_person: tuple[str,str] | None = None
    detected_person: tuple[str,str] | None = None
    retrieved_data: str | None = None
    answer: ThinkingResponse | None = None


class AgentConfig(BaseModel):
    name_surname_list: list[tuple[str,str]]
    pinecone_api_key: str
    pinecone_index_namespace: str
    groq_api_key: str
    llm_model: str    
    llm_temperature: float
    search_top_k: int


class Agent:
    def __init__(self, config: AgentConfig):
        """
        Initializes the Agent with configuration, language model, Pinecone client,
        and constructs the LangGraph workflow.

        This includes:
        - Setting up the LLM (ChatGroq)
        - Initializing Pinecone indexes for each user
        - Building a dynamic graph with:
        - A person detection node
        - User-specific lookup nodes
        - A response generation node
        - Creating conditional routing based on detected person
        - Saving a visualization of the graph as a Mermaid PNG

        Args:
            config (AgentConfig): Configuration object containing API keys, model settings,
                                user list, and retrieval parameters.
        """
        # init llm
        self.llm = ChatGroq(
            api_key=config.groq_api_key, # type: ignore
            model=config.llm_model,
            temperature=config.llm_temperature,
        )

        # init pinecone
        self.pc = Pinecone(api_key=config.pinecone_api_key)

        graph = StateGraph(AgentState)
        graph.add_node("determine_person", self.determine_person)
        graph.add_node("write_response", self.generate_response)
        graph.add_edge("write_response", END)

        self.users = config.name_surname_list

        # build user-specific dispatching
        self.lookups = dict()
        mapping = dict()
        for user in self.users:
            hyphen_name = "-".join(user)
            node_name = f"lookup-{hyphen_name}"
            
            lookup = PineconeLookup(
                index=self.pc.Index(hyphen_name),
                namespace=config.pinecone_index_namespace,
                top_k=config.search_top_k
            )

            # add lookup node and edge for writing response
            graph.add_node(node_name, self.build_lookup(lookup))
            graph.add_edge(node_name, "write_response")

            # prepare conditional edges
            mapping[user] = node_name
            
            # save it
            self.lookups[user] = lookup

        # add conditional edges using dummy function, as determine_person already does the heavy lifting
        graph.add_conditional_edges(
            "determine_person",
            lambda state: mapping[state.detected_person]
        )

        graph.set_entry_point("determine_person")
        self.graph = graph.compile()


    def build_lookup(self, lookup_helper: PineconeLookup):
        """
        Creates a lookup node function for retrieving relevant data.

        Args:
            lookup_helper (PineconeLookup): Helper object to perform vector search.

        Returns:
            Callable[[AgentState], AgentState]: A function that retrieves and injects
            relevant data into the state.
        """
        def look_up_custom(state: AgentState):
            result = lookup_helper.lookup(
                query=state.user_prompt
            )

            return AgentState(retrieved_data=result)
        
        return look_up_custom

    def determine_person(self, state: AgentState):
        """
        Determines which person from the configured user list is referenced
        in the user's query.

        Uses an LLM prompt to classify the query against a numbered list of users.
        If no person is identified, defaults to index 0 (default_person).

        Args:
            state (AgentState): Current state containing the user prompt and default person.

        Returns:
            AgentState: Updated state with the detected_person field set.
        """
        # get user prompt
        user_prompt = state.user_prompt
        
        users = "\n".join(
            f"{idx+1}) {name} {surname}" 
            for idx,(name, surname) in enumerate(self.users)
        )
        
        sys_prompt = PERSON_DETECTION_PROMPT.format(
            users=users,
            user_prompt=user_prompt
        )

        raw_response = self.llm.invoke([
            SystemMessage(content=sys_prompt), 
            HumanMessage(content=user_prompt)
        ])
        
        response = ThinkingResponse.from_llm_output(raw_response.content)

        # extract number
        detected_index = int(response.answer)

        options = [state.default_person] + self.users

        return AgentState(detected_person=options[detected_index])
    
    def generate_response(self, state: AgentState):
        """
        Writes a final answer to the user's question based on retrieved CV data.

        Args:
            state (AgentState): Current state containing the user prompt and retrieved data.

        Returns:
            AgentState: Updated state with the generated ThinkingResponse.
        """
        sys_prompt = ANSWER_PROMPT.format(
            user_prompt=state.user_prompt,
            retrieved_data=state.retrieved_data
        )

        raw_response = self.llm.invoke([
            SystemMessage(content=sys_prompt), 
            HumanMessage(content=state.user_prompt)
        ])
        
        response = ThinkingResponse.from_llm_output(raw_response.content)

        return AgentState(answer=response)

    def answer(self, question: str, user: list[str,str]) -> AgentState:
        """
        Executes the full agent pipeline to answer a user's question.

        This method initializes the agent state with the user query and default person,
        then runs the LangGraph workflow, which includes:
        - Person detection
        - Data retrieval from Pinecone
        - Answer generation

        Args:
            question (str): The user's query.
            user (tuple[str, str]): Default person (name, surname) to use if no match is detected.

        Returns:
            AgentState: Final state containing detected person, retrieved data, and answer.
        """

        result = self.graph.invoke(AgentState(
            user_prompt=question,
            default_person=user
        ))

        return AgentState(**result)
