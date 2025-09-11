from flask import Flask, render_template, request
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Custom helpers
from src.helper import load_embeddings_model, load_pdf_files, text_split
from src.prompt import system_prompt

# Flask app
app = Flask(__name__)
load_dotenv()

# API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# Load and process PDFs
docs = load_pdf_files("data")          # Folder with PDFs
text_chunks = text_split(docs)         # Split into smaller chunks

# Load embeddings
embeddings = load_embeddings_model()

# Pinecone vector store
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Cohere chat model
chat_model = ChatCohere(
    model="command-r",
    temperature=0,
    api_key=COHERE_API_KEY
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Chains
qa_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# Flask routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
