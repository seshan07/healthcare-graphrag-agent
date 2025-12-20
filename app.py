import pickle
import networkx as nx
from pyvis.network import Network

from src.rag_agent import HealthcareRAGAgent

GRAPH_PATH = "data/graph/healthcare_graph.pkl"


class GraphRAGQueryEngine:
    def __init__(self):
        # Load agent (handles vector search + local answer generation)
        self.agent = HealthcareRAGAgent()

        # Load knowledge graph
        with open(GRAPH_PATH, "rb") as f:
            self.graph = pickle.load(f)

    #Direct Q&A
    def direct_qa(self, question: str):
        return self.agent.run(question)

    #Multi-hop reasoning using graph paths
    def multi_hop_reasoning(self, source: str, target: str):
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return []

    #Answer + reasoning path
    def answer_with_reasoning(self, question: str):
        if "insulin" in question.lower() and "complication" in question.lower():
            path = self.multi_hop_reasoning("Insulin", "Complications")
            answer = self.agent.run(question)

            return {
                "answer": answer,
                "path": path
            }
        else:
            return {
                "answer": self.direct_qa(question),
                "path": []
            }

    #Graph visualization with PyVis
    def visualize_graph(self, highlight_path=None, output_file="graph.html"):
        net = Network(height="600px", width="100%", directed=True)

        # Add nodes
        for node in self.graph.nodes:
            net.add_node(node, label=node)

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            label = data.get("relationship", "")
            color = "red" if highlight_path and u in highlight_path and v in highlight_path else "gray"
            net.add_edge(u, v, label=label, color=color)

        net.write_html(output_file, open_browser=False)
        return output_file


if __name__ == "__main__":
    engine = GraphRAGQueryEngine()

    #Direct Q&A
    print(engine.direct_qa("How does insulin work?"))

    print("\n" + "-" * 50 + "\n")

    #Multi-hop reasoning
    result = engine.answer_with_reasoning(
        "How does insulin affect complications through blood glucose?"
    )

    print("Answer:")
    print(result["answer"])

    print("\nReasoning Path:")
    print(" → ".join(result["path"]) if result["path"] else "No path found")

    #Graph visualization
    engine.visualize_graph(highlight_path=result["path"])
