Healthcare GraphRAG Agent 

Problem Statement
Healthcare information related to chronic diseases like diabetes is scattered across long documents such as guidelines, patient education PDFs, and clinical articles. Traditional keyword search or basic RAG systems struggle with multi-step reasoning, concept relationships, and causal understanding.
 
This project solves that problem by combining vector-based retrieval with a healthcare knowledge graph, enabling structured, multi-hop reasoning over medical concepts.

Why Healthcare and Diabetes
Diabetes is a chronic condition with clear relationships between concepts such as insulin, blood glucose, complications, diet, and exercise. This makes it an ideal domain to demonstrate GraphRAG, where understanding relationships is as important as retrieving relevant text.

The dataset consists of real healthcare PDFs covering diabetes types, complications, management, and prevention.

High-Level Architecture
Raw healthcare documents are ingested and converted into text.
Text is split into manageable chunks.
Each chunk is converted into embeddings and stored in FAISS.
An LLM extracts medical entities and relationships from chunks.
Entities and relationships are stored as a directed knowledge graph using NetworkX.
An agent orchestrates graph search, vector search, and answer generation.
A Streamlit UI allows users to query the system and visualize the knowledge graph.

How GraphRAG Works in This Project
Vector retrieval is used to find semantically relevant text chunks.
A knowledge graph is used to model relationships between medical concepts.
During a query, the system uses the graph to expand context through related nodes.
This combined context is passed to the LLM to generate more accurate and explainable answers.

Agent Workflow
User enters a healthcare question.
The agent classifies the intent of the question.
Relevant nodes are identified from the knowledge graph.
Related document chunks are retrieved from FAISS.
Context is assembled from both graph and vector results.
The LLM generates a final answer grounded in retrieved evidence.

Project Structure
data contains raw healthcare documents.
src contains ingestion, graph construction, agent logic, and query engine.
app.py contains the Streamlit user interface.
requirements.txt lists all dependencies.

How to Run Locally
Create a Python virtual environment.
Install dependencies using pip install -r requirements.txt.
Set your OpenAI API key as an environment variable.
Run the ingestion pipeline to create embeddings.
Run the Streamlit app using streamlit run app.py.

Sample Queries
How does insulin control blood glucose.
How does insulin affect complications through blood glucose.
What are the complications caused by diabetes.
How does exercise help in diabetes management.

Graph Visualization
The application visualizes the healthcare knowledge graph.
Nodes represent medical concepts.
Edges represent relationships such as causes, controls, or reduces.
Relevant paths are highlighted during multi-hop reasoning queries.

Limitations
Entity and relationship extraction is LLM-based and not medically perfect.
The knowledge graph is relatively small and domain-specific.
The system is designed for demonstration and learning, not clinical decision-making.

Future Improvements
Improve entity extraction with medical NLP models.
Add support for more diseases and conditions.
Enhance graph reasoning strategies.
Persist vector and graph stores using cloud storage.

This project demonstrates how GraphRAG can be applied to healthcare to enable explainable, multi-hop reasoning over complex medical knowledge.
