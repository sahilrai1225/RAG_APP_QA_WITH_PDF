import streamlit as st
import os
from dotenv import load_dotenv

# LCEL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Memory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Docs + Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Retrieval
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# LLM
from langchain_groq import ChatGroq

# ------------------ ENV ------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI PDF Assistant", layout="wide")

# ------------------ UI ------------------
st.markdown("## 📄 AI PDF Assistant ")

with st.sidebar:
    session_id = st.text_input("Session ID", "default")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ------------------ CHECK ------------------
if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY in .env")
    st.stop()

# ------------------ LLM ------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key="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",
    streaming=True
)

# ------------------ EMBEDDINGS ------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ MEMORY ------------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ------------------ PROCESS PDFs ------------------
if uploaded_files:

    documents = []

    for file in uploaded_files:
        with open("temp.pdf", "wb") as f:
            f.write(file.getvalue())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        if not docs:
            st.warning(f"No text found in {file.name}")
            continue

        for d in docs:
            if d.page_content.strip():
                d.metadata["source"] = file.name
                documents.append(d)

    if not documents:
        st.error("❌ No readable text found. Use text-based PDFs.")
        st.stop()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    splits = [s for s in splits if s.page_content.strip()]

    if not splits:
        st.error("❌ No usable text chunks found.")
        st.stop()

    # ------------------ RETRIEVERS ------------------

    faiss_store = FAISS.from_documents(splits, embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 4

    def hybrid_search(query):
        dense = faiss_retriever.invoke(query)
        sparse = bm25_retriever.invoke(query)

        seen = set()
        results = []

        for d in dense + sparse:
            key = d.page_content
            if key not in seen:
                seen.add(key)
                results.append(d)

        return results[:6]

    # ------------------ PROMPTS ------------------

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using context.\n"
         "Cite like [Page X - filename].\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    rewrite_chain = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
    }
    | rewrite_prompt
    | llm
    | StrOutputParser()
)
    retrieval_chain = rewrite_chain | RunnableLambda(hybrid_search)

    def format_docs(docs):
        return "\n\n".join([
            f"[Page {d.metadata.get('page')} - {d.metadata.get('source')}]\n{d.page_content[:300]}"
            for d in docs
        ])

    rag_chain = (
    {
        "context": retrieval_chain | format_docs,
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

    chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # ------------------ PDF VIEW ------------------
    st.subheader("📄 PDF Preview")

    with open("temp.pdf", "rb") as f:
        st.download_button("Download PDF", f, file_name="preview.pdf")

    # ------------------ CHAT ------------------
    history = get_session_history(session_id)

    for msg in history.messages:
        role = "🧑" if msg.type == "human" else "🤖"
        st.write(f"{role} {msg.content}")

    user_input = st.chat_input("Ask your PDF...")

    if user_input:
        st.write(f"🧑 {user_input}")

        placeholder = st.empty()
        full = ""

        retrieved_docs = hybrid_search(user_input)

        for chunk in chain.stream(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        ):
            full += chunk
            placeholder.write(f"🤖 {full}")

        # ------------------ SOURCES ------------------
        st.subheader("📚 Sources")

        for d in retrieved_docs:
            st.markdown(f"**{d.metadata.get('source')} | Page {d.metadata.get('page')}**")
            st.write(d.page_content[:200])

else:
    st.info("👈 Upload PDFs from sidebar to start")