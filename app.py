import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage

# API Key setup
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else "AIzaSyBNpdl-PxSmwDnM4qbR6i5cfR-NqrzObm4"

# Create chroma_db directory if not exists
CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

st.title("📄 SBI Life Ria 2.0 – Your Insurance Assistant")

# Upload PDFs
uploaded_files = st.file_uploader("Upload SBI Life Brochures (PDFs)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Uploaded {file.name}")
        loader = PyPDFLoader(file.name)
        all_docs.extend(loader.load())

    # Split + Embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    retriever = vectorstore.as_retriever()

    # Setup LLM
    system_prompt = (
        "You are Ria 2.0, a very friendly and helpful SBI Life insurance assistant. "
        "Your goal is to help users, especially beginners, find the most suitable life insurance policy. "
        "Keep all responses under 500 words and use simple, clear language. Be polite and supportive. "
        "You are an expert in all SBI Life policies from brochures provided. Greet the user first."
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        system_message=SystemMessage(content=system_prompt)
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.header("👤 Let's Build Your Insurance Profile")
    age = st.text_input("What is your age?")
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widowed"])
    children = st.selectbox("Do you have children?", ["Yes", "No"])
    goal = st.selectbox("What is your goal?", ["Savings", "Protection", "Retirement", "Child Plan"])
    risk = st.selectbox("Risk Preference", ["Low", "Medium", "High"])

    if st.button("🔍 Get Policy Recommendation"):
        profile_summary = (
            f"The user is {age} years old, {marital_status.lower()}, has children: {children}, "
            f"goal: {goal}, and a {risk.lower()} risk tolerance. "
            f"Based on the brochures you have access to, suggest the most suitable SBI Life policy for this user. "
            f"Keep your answer clear, under 500 words, and friendly."
        )
        response = qa.run(profile_summary)
        st.subheader("✅ Recommended Policy")
        st.write(response)

    st.header("💬 Ask More Questions")
    user_question = st.text_input("Ask Ria 2.0 anything about SBI Life policies:")
    if st.button("📨 Submit Question"):
        if user_question.strip():
            answer = qa.run(user_question)
            st.write(answer)
        else:
            st.warning("Please enter a question.")
