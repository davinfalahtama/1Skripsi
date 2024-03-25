import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_path):
    text = ""
    for file_path in pdf_path:
        if file_path.endswith('.pdf'):
            # Process PDF file
            pdf_reader = PdfReader(file_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualize_system_prompt():
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5,convert_system_message_to_human=True)
    contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()
    return contextualize_q_chain

def contextualized_question(input: dict):
    if input.get("chat_history"):
        contextualize_q_chain = contextualize_system_prompt()
        return contextualize_q_chain
    else:
        return input["question"]

def get_conversational_chain():
    prompt_template = """
        You are a personal Bot assistant for answering any questions about documents of given documents.\n
        You are given a question and a set of documents.\n
        If the user's question requires you to provide specific information from the documents, give your answer based only on the examples provided below. DON'T generate an answer that is NOT written in the provided examples.\n
        If you don't find the answer to the user's question with the examples provided to you below, answer that you didn't find the answer in the documentation and propose him to rephrase his query with more details.\n
        Use bullet points if you have to make a list, only if necessary.\n
        If the question is about code, answer that you don't know the answer.\n
        If the user ask about your name, answer that your name is Elena.\n
        DO NOT EVER ANSWER QUESTIONS THAT IS NOT IN THE DOCUMENTS!\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5,convert_system_message_to_human=True)

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | model
    )    
    return rag_chain

def response_generator(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

def main():
    st.set_page_config(
    page_title="Chat Documents",
    page_icon="🧊",
   # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an extremely cool app!"
    })
    st.header(':sparkles: Mau nanya tentang PMB ITPLN :question:', divider='rainbow')
    st.subheader("Hallo, aku Elena. Temukan informasi seputar PMB ITPLN bersamaku.")
    with st.chat_message("assistant"):
                st.markdown("Kamu mau nanya apa?")
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    docs_path = "docs"
    pdf_docs = [os.path.join(docs_path, filename) for filename in os.listdir(docs_path) if filename.endswith('.pdf')]
    
    with st.spinner("Processing..."):
        start_processing_time = time.time()  # Catat waktu awal pemrosesan
        
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        
        end_processing_time = time.time()  # Catat waktu akhir pemrosesan
        processing_time = end_processing_time - start_processing_time  # Hitung waktu pemrosesan
        st.info(f"PDF files processed successfully in {processing_time:.2f} seconds.")  # Tampilkan waktu pemrosesan
    
    for message in st.session_state["chat_history"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    
    # Accept user input
    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        start_inference_time = time.time()  # Catat waktu awal inferensi
        
        rag_chain = get_conversational_chain()
        ai_msg = rag_chain.invoke({"question": prompt, "chat_history": st.session_state["chat_history"]})

        end_inference_time = time.time()  # Catat waktu akhir inferensi
        inference_time = end_inference_time - start_inference_time  # Hitung waktu inferensi
        
        with st.chat_message("assistant"):
            # Menampilkan markdown langsung
           # response_generator_output = response_generator(ai_msg.content)
            st.markdown(ai_msg.content)
            st.info(f"Inference time: {inference_time:.2f} seconds.")  # Tampilkan waktu inferensi
        
        st.session_state["chat_history"].extend([HumanMessage(content=prompt), ai_msg])

if __name__ == "__main__":
    main()
