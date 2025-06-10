import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage

st.set_page_config(page_title="Ria 2.0 - SBI Life Assistant")

st.title("👩‍💼 Ria 2.0 – SBI Life Insurance Assistant")

# Upload PDFs
uploaded_files = st.file_uploader("Upload SBI Life Policy PDFs", type=["pdf"], accept_multiple_files=True)

# API Key input
api_key = st.text_input("🔑 Enter your Google API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

if uploaded_files and api_key:
    all_docs = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(uploaded_file.name)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are Ria 2.0, a helpful and friendly SBI Life assistant..."
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        system_message=SystemMessage(content=system_prompt)
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ Documents loaded and vector store ready!")

    # Profile Inputs
    with st.form("user_profile"):
        st.subheader("📋 Tell me a bit about yourself")
        age = st.text_input("Your age")
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Widowed"])
        children = st.selectbox("Do you have children?", ["Yes", "No"])
        goal = st.selectbox("Goal", ["Savings", "Protection", "Retirement", "Child Plan"])
        risk = st.selectbox("Risk preference", ["Low", "Medium", "High"])
        submitted = st.form_submit_button("Get Policy Recommendation")

    if submitted:
        summary = (
            f"The user is {age} years old, {marital_status}, has children: {children}, "
            f"goal: {goal}, and a {risk} risk tolerance..."
        )
        response = qa.run(summary)
        st.subheader("✅ Recommended Policy")
        st.write(response)

    # Q&A Loop
    st.subheader("💬 Ask a Follow-up")
    query = st.text_input("Type your question")
    if query:
        result = qa.run(query)
        st.write(result)
