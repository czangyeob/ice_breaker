import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


def ingest_docs():
    loader = ReadTheDocsLoader(
        path="langchain-docs/"
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Goint to insert {len(documents)} to Pinecone")

    # embeddings = OpenAIEmbeddings()
    # Pinecone.from_documents(documents=documents, embedding=embeddings, name='langchain-docs-index')
    print("****** Added to Pinecone vectorstore ")


if __name__ == "__main__":
    ingest_docs()
