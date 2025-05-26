import os
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

load_dotenv()

# ---- TOOL DEFINITIONS ----
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two integers."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers. Raises error on divide-by-zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Compute the modulus of two integers."""
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia and return top 2 results."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return "\n\n---\n\n".join(doc.page_content[:1000] for doc in search_docs)

@tool
def web_search(query: str) -> str:
    """Search the web using Tavily and return top 3 results."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    return "\n\n---\n\n".join(doc.page_content[:1000] for doc in search_docs)

@tool
def arxiv_search(query: str) -> str:
    """Search arXiv.org and return top 3 documents."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    return "\n\n---\n\n".join(doc.page_content[:1000] for doc in search_docs)

# ---- SYSTEM PROMPT ----
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()
sys_msg = SystemMessage(content=system_prompt)

# ---- FAISS RETRIEVER ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")

if not os.path.exists(index_path):
    print("FAISS index not found. Building new index...")
    os.makedirs(index_path, exist_ok=True)
    loader = TextLoader("corpus.txt")
    docs = loader.load()
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
else:
    print("FAISS index found. Loading index...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

retriever_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)

# ---- TOOL LIST ----
tools = [
    multiply, add, subtract, divide, modulus,
    wiki_search, web_search, arxiv_search,
    retriever_tool
]

# ---- GRAPH CONSTRUCTION ----
def build_graph():
    provider = os.getenv("LLM_PROVIDER", "huggingface")

    if provider == "google":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="HuggingFaceH4/zephyr-7b-beta",
                temperature=0,
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
        )
    else:
        raise ValueError("Choose either 'google' or 'huggingface' as LLM_PROVIDER.")

    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])], "next": "check"}

    def retriever(state: MessagesState):
        similar = vectorstore.similarity_search(state["messages"][0].content)
        if similar:
            example = HumanMessage(content=f"Here is a similar question and answer:\n\n{similar[0].page_content}")
            return {"messages": [sys_msg] + state["messages"] + [example]}
        return {"messages": [sys_msg] + state["messages"]}

    def check_if_final(state: MessagesState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not hasattr(last, "tool_calls"):
            return "end"
        return "tools"

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_conditional_edges("assistant", check_if_final)
    builder.set_entry_point("retriever")
    builder.set_finish_point("assistant")
    builder.add_edge("retriever", "assistant")
    builder.add_edge("tools", "assistant")
    return builder.compile()

def append_to_corpus(question: str, answer: str, corpus_path: str = "corpus.txt") -> None:
    formatted = f"{question.strip()}\n{answer.strip()}\n\n"
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            if formatted in f.read():
                return
    with open(corpus_path, "a", encoding="utf-8") as f:
        f.write(formatted)

if __name__ == "__main__":
    graph = build_graph()
    q = "What is 15 divided by 3?"
    messages = [HumanMessage(content=q)]
    result = graph.invoke({"messages": messages})
    for m in result["messages"]:
        m.pretty_print()