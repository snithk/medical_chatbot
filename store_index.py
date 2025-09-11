from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_tp_minimal_docs, text_split, load_embeddings_model

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")   # OpenAI
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")   # Cohere (if you want to support it too)

# Load and preprocess documents
extracted= load_pdf_files("data")
filtered_docs = filter_tp_minimal_docs(extracted)
text_chunks=text_split(filtered_docs)

# Load embeddings

embeddings = load_embeddings_model()

# Initialize Pinecone client

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

pc = Pinecone(api_key="pcsk_6NrbQW_LinFsGaW8vvdQLaWPnPZXpivZpz4KY9zb1PFq7F7RiHUqUJBdnv5SDg3psoaxzq")  

index_name = "medical-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
# Store docs in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
