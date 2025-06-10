# app.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage

# Title
st.set_page_config(page_title="Ria 2.0 – SBI Life Assistant", page_icon="🤖")
st.title("👋 Ria 2.0 – SBI Life Insurance Assistant")

# Step 3: API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBNpdl-PxSmwDnM4qbR6i5cfR-NqrzObm4"

# Step 4: Upload PDFs
st.sidebar.header("📄 Upload SBI Life Brochures")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔍 Loading and processing PDFs..."):
        all_docs = []
        for file in uploaded_files:
            loader = PyPDFLoader(file)
            all_docs.extend(loader.load())

        # Step 5: Split and Embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(all_docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Step 6: LLM Setup
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

        st.success("✅ Documents processed successfully!")

        st.subheader("📝 Tell me about yourself:")

        # Step 7: User Profile Form
        with st.form("user_profile"):
            age = st.text_input("What is your age?")
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widowed"])
            children = st.selectbox("Do you have children?", ["Yes", "No"])
            goal = st.selectbox("Goal", ["Savings", "Protection", "Retirement", "Child Plan"])
            risk = st.selectbox("Risk Preference", ["Low", "Medium", "High"])
            submitted = st.form_submit_button("Get Policy Recommendation")

        if submitted:
            profile_summary = (
                f"The user is {age} years old, {marital_status.lower()}, has children: {children}, "
                f"goal: {goal}, and a {risk.lower()} risk tolerance. "
                f"Based on the brochures you have access to, suggest the most suitable SBI Life policy for this user. "
                f"Keep your answer clear, under 500 words, and friendly."
            )

            with st.spinner("💡 Analyzing your profile and finding the best policy..."):
                response = qa.run(profile_summary)

            st.subheader("✅ Recommended Policy")
            st.write(response)

            # Step 8: Chat Q&A
            st.subheader("💬 Ask Follow-up Questions")
            user_question = st.text_input("Ask anything about SBI Life policies:")

            if user_question:
                follow_up = qa.run(user_question)
                st.write(f"**Ria 2.0:** {follow_up}")

else:
    st.info("📥 Please upload one or more SBI Life brochure PDFs to begin.")
