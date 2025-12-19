import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


DATA_DIR = "data"
VECTOR_DB_DIR = "vector_store"


def load_documents():
    """
    Load all PDF documents from the data directory
    """
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def split_documents(documents):
    """
    Split documents into chunks of 300–500 words
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    """
    Generate embeddings and store them in FAISS
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_DIR)


def main():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)

    print("Creating vector store...")
    create_vector_store(chunks)

    print("Ingestion complete. Vector store saved.")


if __name__ == "__main__":
    main()

