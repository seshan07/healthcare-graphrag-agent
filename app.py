import streamlit as st
from src.rag_agent import run_agent
from src.query_engine import direct_qa, multi_hop_reasoning, visualize_graph

st.set_page_config(layout="wide")

st.title("Healthcare GraphRAG Agent")

question = st.text_input("Ask a healthcare question")

ask = st.button("Ask")

if ask and question:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Answer")
        answer = run_agent(question)
        st.write(answer)

    with col2:
        st.subheader("Knowledge Graph")
        graph_file = visualize_graph()
        with open(graph_file, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=600, scrolling=True)
