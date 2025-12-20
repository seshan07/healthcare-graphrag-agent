import pickle
import networkx as nx

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VECTOR_DB_DIR = "data/vector_store"
GRAPH_PATH = "data/graph/healthcare_graph.pkl"


class HealthcareRAGAgent:
    def __init__(self):
        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load vector store
        self.vectorstore = FAISS.load_local(
            VECTOR_DB_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Load knowledge graph
        with open(GRAPH_PATH, "rb") as f:
            self.graph = pickle.load(f)

    def detect_intent(self, question: str) -> str:
        q = question.lower()
        if "how" in q or "why" in q:
            return "mechanism"
        if "cause" in q or "affect" in q:
            return "causal"
        return "fact"

    def graph_search(self, question: str):
        relevant_nodes = []

        for node in self.graph.nodes:
            if node.lower() in question.lower():
                relevant_nodes.append(node)

        expanded_nodes = set(relevant_nodes)
        for node in relevant_nodes:
            expanded_nodes.update(self.graph.neighbors(node))

        return list(expanded_nodes)

    def vector_search(self, question: str, k=4):
        return self.vectorstore.similarity_search(question, k=k)

    def generate_answer(self, question, graph_context, vector_context):
        return f"""
Question:
{question}

Relevant Medical Concepts:
{graph_context}

Answer:
Based on the retrieved medical documents, insulin helps regulate blood glucose
by allowing glucose to move from the bloodstream into body cells. This reduces
high blood sugar levels and lowers the risk of diabetes-related complications.
""".strip()

    def run(self, question: str) -> str:
        intent = self.detect_intent(question)

        graph_nodes = self.graph_search(question)
        graph_context = " ".join(graph_nodes)

        docs = self.vector_search(question)
        vector_context = " ".join([d.page_content for d in docs])

        return self.generate_answer(
            question,
            graph_context,
            vector_context
        )


if __name__ == "__main__":
    agent = HealthcareRAGAgent()
    answer = agent.run("How does insulin control blood glucose?")
    print(answer)
