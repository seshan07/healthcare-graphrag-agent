from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import networkx as nx

VECTOR_DB_DIR = "vector_store"
GRAPH_PATH = "healthcare_graph.gml"

llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

def load_vector_store():
    return FAISS.load_local(VECTOR_DB_DIR, embeddings)

def load_graph():
    return nx.read_gml(GRAPH_PATH)

def classify_intent(question):
    prompt = "Classify the intent of the question into one word such as mechanism, treatment, cause, prevention, or complication.\nQuestion:\n" + question
    return llm(prompt).strip().lower()

def graph_search(graph, question):
    related_nodes = []
    for node in graph.nodes:
        if node.lower() in question.lower():
            related_nodes.append(node)
            related_nodes.extend(list(graph.successors(node)))
            related_nodes.extend(list(graph.predecessors(node)))
    return list(set(related_nodes))

def vector_search(vector_store, query):
    docs = vector_store.similarity_search(query, k=5)
    return docs

def generate_answer(question, context, intent):
    context_text = "\n".join(context)
    prompt = "Answer the healthcare question based on the context.\nIntent: " + intent + "\nContext:\n" + context_text + "\nQuestion:\n" + question
    return llm(prompt)

def run_agent(question):
    intent = classify_intent(question)
    graph = load_graph()
    vector_store = load_vector_store()
    nodes = graph_search(graph, question)
    graph_context = " ".join(nodes)
    docs = vector_search(vector_store, question + " " + graph_context)
    context = [doc.page_content for doc in docs]
    answer = generate_answer(question, context, intent)
    return answer
