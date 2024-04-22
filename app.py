import streamlit as st
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter



#######################################
# BACKEND

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
        # Initialize the textsplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks



#######################################
# FRONTEND 
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.text_input ("Ask a question about your documentos:")

    with st.sidebar:
        st.subheader("Introduce your API Key")
        azure_api_key = st.text_input("Azure API Key", type="password")
        #st.text_input("OpenAI API Key", type="password")
        st.divider()
        ####
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF
                raw_text = get_pdf_text(pdf_docs)
                                
                # Get Text Chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)


                # Create vectorstore



if __name__ == '__main__':
    main()