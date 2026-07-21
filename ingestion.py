import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # ponytail: needed before module-level from_existing_index

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
)

# ponytail: read-only at import; upsert only via `python ingestion.py`
retriever = PineconeVectorStore.from_existing_index(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
).as_retriever()


if __name__ == "__main__":
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs_list = [doc for url in urls for doc in WebBaseLoader(url).load()]
    doc_splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    ).split_documents(docs_list)

    PineconeVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"],
    )
    print(f"Upserted {len(doc_splits)} chunks into {os.environ['INDEX_NAME']}")
