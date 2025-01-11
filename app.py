import os
import io
from gtts import gTTS
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import pandas as pd  # For saving responses in .xlsx format
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    """Load a PDF document using PyMuPDF and extract text content.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of Document objects containing extracted text.
    """
    with fitz.open(file_path) as doc:
        text = "".join([page.get_text("text") for page in doc])
    return [Document(page_content=text)]

def setup_vectorstore(documents):
    """Create a FAISS vector store from the given documents.

    Args:
        documents (list): List of Document objects.

    Returns:
        FAISS: Initialized FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(doc_chunks, embeddings)

def get_response(vectorstore, question, chat_history):
    """Generate a chatbot response using vector store and ChatGroq.

    Args:
        vectorstore (FAISS): FAISS vector store for document search.
        question (str): User query.
        chat_history (list): List of chat history dictionaries.

    Returns:
        tuple: Assistant response and TTS audio stream.
    """
    docs = vectorstore.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])

    full_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    full_prompt += f"\nUser: {question}\nContext: {context}\nAssistant:"

    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    response = llm.invoke(full_prompt)
    assistant_response = response.content

    tts = gTTS(assistant_response, lang="en")
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)

    return assistant_response, audio_fp

def save_chat_history(chat_history, file_type="txt"):
    """Save chat history to a file.

    Args:
        chat_history (list): List of chat messages.
        file_type (str): File type to save ('txt', 'pdf', or 'xlsx').
    """
    if file_type == "txt":
        with open("chat_history.txt", "w") as f:
            for msg in chat_history:
                f.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
    elif file_type == "xlsx":
        df = pd.DataFrame(chat_history)
        df.to_excel("chat_history.xlsx", index=False)
    elif file_type == "pdf":
        from reportlab.pdfgen import canvas
        c = canvas.Canvas("chat_history.pdf")
        y = 800
        for msg in chat_history:
            c.drawString(50, y, f"{msg['role'].capitalize()}: {msg['content']}")
            y -= 20
        c.save()

# Streamlit setup
st.set_page_config(page_title="Chat with Doc", page_icon="📄", layout="centered")
st.title("🦙 Chat with Doc - LLAMA 3.1")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstore" not in st.session_state:
        documents = load_document(file_path)
        st.session_state.vectorstore = setup_vectorstore(documents)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        assistant_response, audio_fp = get_response(st.session_state.vectorstore, user_input, st.session_state.chat_history)
        st.markdown(assistant_response)
        st.audio(audio_fp, format="audio/mp3")
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

if st.button("Save Chat History"):
    file_type = st.selectbox("Select file type", ["txt", "pdf", "xlsx"])
    save_chat_history(st.session_state.chat_history, file_type)
    st.success(f"Chat history saved as 'chat_history.{file_type}'!")
