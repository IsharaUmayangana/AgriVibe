from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage

from agrivibe.agents.fieldoracle.schemas import CitedAnswer
from agrivibe.models.ask import AskResponse
from agrivibe.models.knowledge import Source


def to_source(doc: Document) -> Source:
    metadata = doc.metadata
    return Source(
        id=doc.id,
        source=metadata["source"],
        page=metadata["page"],
        category=metadata["category"],
        text=doc.page_content
    )


def retrieved_chunks(messages: list[BaseMessage]) -> dict[str, Document]:
    """Collects every chunk the search tool returned during one agent run.
    
    Args:
        messages: The agent's message history for the run.

    Returns:
        The retrieved chunks, keyed by chunk id.

    """

    chunks = {}

    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_knowledge":
            for doc in message.artifact or []:
                chunks[doc.id] = doc

    return chunks


def search_queries(messages: list[BaseMessage]) -> list[str]:

    """Lists the queries the agent chose to search for.
    
    
    Args:
        messages: The agent's message history for the run.

    Returns:
        The search queries, in the order the agent ran them.
    """

    return(
        call["args"]["query"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "search_knowledge"
    )




class AnswerService:
    def __init__(self, rag_agent):
        """Initializes the service.
        
        Args:
            rag_agent: The agent built by 'build_rag_agent'.
        """

        self.rag_agent = rag_agent


    def ask(self, question: str):

        result = self.rag_agent.invoke({"messages": [{"role": "user", "content": question}]})

        answer: CitedAnswer =  result["structured_response"]
        messages = result["messages"]

        found = answer.found
        sentences = answer.sentences
        cited = {chunk_id for s in sentences for chunk_id in s.citations}

        return AskResponse(
            question=question,
            found=found,
            answer= " ".join(f"{s.sentence} [{', '.join(s.citations)}]" for s in sentences),
            sentences=sentences,
            sources= [to_source(doc) for chunk_id, doc in retrieved_chunks(messages).items() if chunk_id in cited],
            searches=search_queries(messages)
        )