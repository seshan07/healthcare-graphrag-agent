from pyvis.network import Network
import networkx as nx
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

VECTOR_DB_DIR = "vector_store"
GRAPH_PATH = "healthcare_graph.gml"

llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

def load_vector_store():
    return FAISS.load_local(VECTOR_DB_DIR, embeddings)

def load_graph():
    return nx.read_gml(GRAPH_PATH)

def direct_qa(question):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(question, k=5)
    context = "\n".join([d.page_content for d in docs])
    prompt = "Answer the healthcare question using the context.\nContext:\n" + context + "\nQuestion:\n" + question
    return llm(prompt)

def multi_hop_reasoning(question):
    graph = load_graph()
    nodes = []
    for node in graph.nodes:
        if node.lower() in question.lower():
            nodes.append(node)
    paths = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                try:
                    path = nx.shortest_path(graph, nodes[i], nodes[j])
                    paths.append(path)
                except:
                    pass
    vector_store = load_vector_store()
    context_chunks = []
    for path in paths:
        path_text = " ".join(path)
        docs = vector_store.similarity_search(path_text, k=2)
        for d in docs:
            context_chunks.append(d.page_content)
    context = "\n".join(context_chunks)
    prompt = "Answer the healthcare question using multi-hop reasoning.\nContext:\n" + context + "\nQuestion:\n" + question
    return llm(prompt)

def visualize_graph(highlight_path=None):
    graph = load_graph()
    net = Network(height="600px", width="100%", directed=True)
    for node in graph.nodes:
        net.add_node(node, label=node)
    for u, v, data in graph.edges(data=True):
        label = data.get("relation", "")
        color = "red" if highlight_path and u in highlight_path and v in highlight_path else "gray"
        net.add_edge(u, v, label=label, color=color)
    net.show("graph.html")
    return "graph.html"
