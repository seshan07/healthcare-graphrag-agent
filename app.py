import streamlit as st
import tempfile
import os

from src.query_engine import GraphRAGQueryEngine

st.set_page_config(page_title="Healthcare GraphRAG", layout="wide")

st.title("🩺 Healthcare GraphRAG Agent")
st.write("Ask healthcare questions and see graph-based reasoning.")

engine = GraphRAGQueryEngine()

question = st.text_input(
    "Ask a healthcare question",
    placeholder="How does insulin affect complications through blood glucose?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = engine.answer_with_reasoning(question)

        st.subheader("Answer")
        st.write(result["answer"])

        if result["path"]:
            st.subheader("Reasoning Path")
            st.write(" → ".join(result["path"]))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                graph_file = engine.visualize_graph(
                    highlight_path=result["path"],
                    output_file=tmp.name
                )

            st.subheader(" Knowledge Graph")
            with open(graph_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

            os.remove(graph_file)
        else:
            st.info("No multi-hop reasoning path found for this question.")
