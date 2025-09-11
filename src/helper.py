from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List
from langchain.schema import Document


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents
def filter_tp_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}  # fixed typo and dictionary syntax
            )
        )
    return minimal_docs

def text_split(filtered_docs):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts_chunk=text_splitter.split_documents(filtered_docs)
    return texts_chunk

def load_embeddings_model():
    """
    Load and return HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings_model

embeddings = load_embeddings_model()