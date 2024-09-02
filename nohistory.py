import streamlit as st
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
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from PyPDF2 import PdfReader

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_path):
    text = ""
    for file_path in pdf_path:
        if file_path.endswith('.pdf'):
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

def get_conversational_chain():
    prompt_template = """ System:
        You are a personal Bot assistant for answering any questions about certain context of given context.\n
        You are given a question and a set of context.\n 
        You are supposed to answer in either Bahasa Indonesia or English, following the language of the user.\n
        If the user's question requires you to provide specific information from the context, give your answer based only on the examples provided below. 
        DON'T generate an answer that is NOT written in the provided examples.\n
        If you don't find the answer to the user's question with the examples provided to you below, 
        answer that you didn't find the answer in the context given and propose him to rephrase his query with more details.\n
        Use bullet points if you have to make a list, only if necessary.\n
        If the question is about code, answer that you don't know the answer.\n
        If there are links available, then you can proceed to access it.\n
        If the user asks about your name, answer that your name is Elena.\n
        If you don't find the answer to the user's question, just say that you don't know.\n
        If the question is about anything that is NOT related to the given context, answer that you don't know the answer.\n
        If there is any inappropriate question, just say that you can't answer the question.\n
        Please give the references from which line or paragraph regarding your answer in the context given.\n
        If the user asks you about generating images or sound, state that you can't do that, you can only generate text as an output.\n
        DO NOT EVER ANSWER QUESTIONS THAT ARE NOT IN THE GIVEN CONTEXT!\n\n

        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=retriever | format_docs
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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an extremely cool app!"
    })
    st.header(':sparkles: Mau nanya tentang PMB ITPLN :question:', divider='rainbow')
    st.subheader("Hallo, aku Elena. Temukan informasi seputar PMB ITPLN bersamaku.")
    with st.chat_message("assistant"):
        st.write_stream(response_generator("Kamu mau nanya apa?"))

    docs_path = "docs"
    pdf_docs = [os.path.join(docs_path, filename) for filename in os.listdir(docs_path) if filename.endswith('.pdf')]
    
    with st.spinner("Processing..."):
        start_processing_time = time.time()
        
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        
        end_processing_time = time.time()
        processing_time = end_processing_time - start_processing_time
        st.info(f"PDF processed successfully in {processing_time:.2f} seconds.")
    
    # Accept user input
    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        start_inference_time = time.time()
        
        rag_chain = get_conversational_chain()
        ai_msg = rag_chain.invoke({"question": prompt})

        end_inference_time = time.time()
        inference_time = end_inference_time - start_inference_time
        
        with st.chat_message("assistant"):
            st.markdown(ai_msg.content)
            st.info(f"Inference time: {inference_time:.2f} seconds.")

if __name__ == "__main__":
    main()