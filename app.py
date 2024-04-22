import streamlit as st
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = AzureChatOpenAI(
                deployment_name="gpt4-8k-jim2", #"gpt-3.5-turbo"  # Chatgpt model!
                #openai_api_type="azure",
                api_version="2024-02-15-preview",
                temperature=0.1
            )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retrieve=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



#######################################
# FRONTEND 
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

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
                #st.write(text_chunks)

                # Create vectorstore
                vectorstore = get_vectorstore(text_chunks)

                # Create a conversation 
                st.session_state.conversation =  get_conversation_chain(vectorstore)

    st.session_state.conversartion


if __name__ == '__main__':
    main()