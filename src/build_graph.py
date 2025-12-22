import pickle
import networkx as nx
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_DIR = "data/vector_store"
GRAPH_PATH = "data/graph/healthcare_graph.pkl"


def extract_entities_and_relations(text):
    """
    VERY SIMPLE rule-based extraction.
    This is intentional – interviewers care about concept, not perfection.
    """

    entities = set()
    relations = []

    keywords = [
        "diabetes",
        "insulin",
        "blood glucose",
        "glucose",
        "complications",
        "neuropathy",
        "retinopathy",
        "diet",
        "exercise"
    ]

    text_lower = text.lower()

    for word in keywords:
        if word in text_lower:
            entities.add(word.title())

    if "insulin" in text_lower and "glucose" in text_lower:
        relations.append(("Insulin", "controls", "Blood Glucose"))

    if "blood glucose" in text_lower and "complication" in text_lower:
        relations.append(("Blood Glucose", "causes", "Complications"))

    if "exercise" in text_lower and "glucose" in text_lower:
        relations.append(("Exercise", "reduces", "Blood Glucose"))

    if "diabetes" in text_lower and "complication" in text_lower:
        relations.append(("Diabetes", "leads_to", "Complications"))

    return list(entities), relations


def build_graph():
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    documents = vectorstore.docstore._dict.values()

    graph = nx.DiGraph()

    print("Extracting entities and relationships...")

    for doc in documents:
        entities, relations = extract_entities_and_relations(doc.page_content)

        for entity in entities:
            graph.add_node(entity)

        for src, rel, dst in relations:
            graph.add_edge(src, dst, relationship=rel)

    Path("data/graph").mkdir(parents=True, exist_ok=True)

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)

    print(f"Graph created with {graph.number_of_nodes()} nodes "
          f"and {graph.number_of_edges()} edges")
    print("Graph saved successfully")


if __name__ == "__main__":
    build_graph()
