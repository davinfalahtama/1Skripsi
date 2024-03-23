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
#=================================================================================
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    #for pdf in pdf_docs:
    pdf_reader= PdfReader(pdf_docs)
    for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

#print(get_pdf_text(r"docs/Panduan Penerimaan Mahasiswa .pdf"))

#fungsinya untuk membaca pdf dan menyatukan seluruh teks yang ada di text="" (string)

#====================================================================================================
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

teks = get_pdf_text(r"docs/Panduan Penerimaan Mahasiswa .pdf")
#print(get_text_chunks(teks))
print(len(get_text_chunks(teks)))

#penjelasan dari kodingan di atas adalah, jadi misal teks pdf kita berjumlah 10k kata, nah teksnya akan dibagi sesuai dengan chunk sizenya misal chunk size 5k
#jadi jika diprint hasilnya adalah 2 karena 10k/5k, nah fungsi overlap yaitu ngambil 500 teks di depannya jadi si 5k kedua ngambil 500 dari 5k pertama tapi si 5k pertama ttp 5k jumlahnhya

#Chunk Size: Ini adalah ukuran maksimum dari setiap potongan teks yang dihasilkan. Dengan kata lain, ini menentukan seberapa besar setiap bagian teks yang dipotong. Misalnya, jika chunk_size diatur ke 1000, maka setiap potongan teks akan memiliki panjang maksimum 1000 karakter.
#Chunk Overlap: Ini adalah seberapa banyak potongan teks yang akan tumpang tindih satu sama lain. Dengan kata lain, ketika satu potongan berakhir, potongan berikutnya akan dimulai sejauh chunk_overlap karakter sebelumnya. Ini memastikan bahwa tidak ada informasi penting yang hilang di antara potongan-potongan tersebut, karena beberapa karakter dari akhir potongan sebelumnya akan dimasukkan ke dalam potongan berikutnya.

#=================================================================================
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

chungz = get_text_chunks(teks)
vectorstore = get_vector_store(chungz)

query = "apa saja fakultas di itpln"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)
#print(vectorstore)
#=================================================================================

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
