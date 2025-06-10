import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Streamlit UI
st.title("🧠 Ria 2.0 - SBI Life Insurance Advisor")

uploaded_files = st.file_uploader("📄 Upload SBI Life policy brochures (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Step 4: Load PDFs
    all_docs = []
    for file in uploaded_files:
        loader = PyPDFLoader(file)
        all_docs.extend(loader.load())

    # Step 5: Split + Embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Step 6: Setup Gemini Chat Model with System Instructions
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

    st.success("📚 Documents processed. Ria 2.0 is ready!")

    # Step 7: Profile-based Recommendation
    st.header("👤 Let's understand your needs")
    with st.form("user_profile"):
        age = st.text_input("What is your age?")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widowed"])
        children = st.selectbox("Do you have children?", ["Yes", "No"])
        goal = st.selectbox("Your goal", ["Savings", "Protection", "Retirement", "Child Plan"])
        risk = st.selectbox("Risk preference", ["Low", "Medium", "High"])
        submitted = st.form_submit_button("Get Policy Recommendation")

    if submitted:
        profile = f"The user is {age} years old, {marital_status}, has children: {children}, goal: {goal}, and a {risk} risk tolerance. Suggest the most suitable SBI Life policy for this user in under 500 words."
        answer = qa.run(profile)
        st.subheader("✅ Recommended Policy")
        st.write(answer)

    # Step 8: Q&A
    st.header("💬 Ask a follow-up question")
    user_q = st.text_input("Ask anything about SBI Life policies:")
    if user_q:
        reply = qa.run(user_q)
        st.write(reply)
