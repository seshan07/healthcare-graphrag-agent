import networkx as nx
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0)

EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Extract medical entities and relationships from the text. Return each relationship in the format: Entity1 | RELATION | Entity2\n\nText:\n{text}"
)

def extract_relationships(text):
    prompt = EXTRACTION_PROMPT.format(text=text)
    response = llm(prompt)
    relations = []
    for line in response.split("\n"):
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                relations.append(parts)
    return relations

def build_knowledge_graph(chunks):
    graph = nx.DiGraph()
    for chunk in chunks:
        relationships = extract_relationships(chunk.page_content)
        for entity1, relation, entity2 in relationships:
            graph.add_node(entity1)
            graph.add_node(entity2)
            graph.add_edge(entity1, entity2, relation=relation)
    return graph

def save_graph(graph, path="healthcare_graph.gml"):
    nx.write_gml(graph, path)

