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

print(get_pdf_text(r"docs/Panduan Penerimaan Mahasiswa .pdf"))

#fungsinya untuk membaca pdf dan menyatukan seluruh teks yang ada di text="" (string)

Gibbeum Lee, Volker Hartmann, Jongho Park,
Dimitris Papailiopoulos, Kangwook Lee