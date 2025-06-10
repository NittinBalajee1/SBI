import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage

# --- API Key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBNpdl-PxSmwDnM4qbR6i5cfR-NqrzObm4"  # Replace with your actual key

st.title("📘 SBI Life RIA 2.0 - Insurance Assistant")
st.markdown("Upload your SBI Life PDF brochures and get policy recommendations based on your profile.")

# --- PDF Upload ---
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔄 Loading documents..."):
        all_docs = []
        for file in uploaded_files:
            loader = PyPDFLoader(file)
            all_docs.extend(loader.load())

    with st.spinner("🔄 Splitting and embedding..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(all_docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Important: Use persist_directory to avoid Chroma runtime errors
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        retriever = vectorstore.as_retriever()

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

    st.success("✅ Documents processed. Let's find your best policy!")

    # --- User Profile Form ---
    with st.form("user_profile_form"):
        age = st.text_input("What is your age?")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widowed"])
        children = st.selectbox("Do you have children?", ["Yes", "No"])
        goal = st.selectbox("What is your goal?", ["Savings", "Protection", "Retirement", "Child Plan"])
        risk = st.selectbox("Risk preference?", ["Low", "Medium", "High"])
        submitted = st.form_submit_button("🔍 Recommend Policy")

    if submitted:
        summary = (
            f"The user is {age} years old, {marital_status}, has children: {children}, "
            f"goal: {goal}, and a {risk} risk tolerance. "
            f"Based on the brochures you have access to, suggest the most suitable SBI Life policy for this user. "
            f"Keep your answer clear, under 500 words, and friendly."
        )
        with st.spinner("🔍 Analyzing and recommending..."):
            response = qa.run(summary)
            st.markdown("### ✅ Recommended Policy:")
            st.write(response)

    # --- Q&A Section ---
    st.markdown("---")
    st.markdown("### 💬 Ask a Follow-up Question")
    user_query = st.text_input("Your question (or leave empty to skip):")
    if user_query:
        with st.spinner("💡 Thinking..."):
            response = qa.run(user_query)
            st.write(response)

