import streamlit as st
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings # from langchain_community.embeddings # from langchain.embeddings 
from langchain_community.vectorstores.faiss import FAISS # from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
#from langchain_community.llms import AzureOpenAI # from langchain.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI # from langchain.chat_models import AzureChatOpenAI < Langachain v0.2.0
from langchain_community.callbacks import get_openai_callback

# BACKEND
def process_tex(text):
    # Initialize the textsplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Load a model to generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Create a FAISS index from the text chunks using embeddings
    knowledge_base = FAISS.from_texts(chunks,embeddings)

    return knowledge_base

def summarizer(pdf):

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        knowledge_base = process_tex(text)

        query = """Eres un asistente legal y te solicitan hacer lo siguiente:
        1. Un resumen general del archivo.
        2. Resumir las ideas y argumentos más importantes del documento.
        3. En caso el documento sea una Ley o Normativa: menciona y enumera cúales son las más importantes.
        """

        if query: 
            docs = knowledge_base.similarity_search(query)

            llm = AzureChatOpenAI(
                deployment_name="gpt4-8k-jim2", #"gpt-3.5-turbo"  # Chatgpt model!
                #openai_api_type="azure",
                api_version="2024-02-15-preview",
                temperature=0.1
            )
            
            chain = load_qa_chain(llm, chain_type='stuff')

            response = chain.run(input_documents=docs, question=query) # Need to modify the chain.run to chain.invoke, but how?

            return response

# FRONTEND 
def main():
    st.set_page_config(page_title="The PDF Summarizer")

    st.title("PDF Summarizing App")
    st.write("Summarize your PDF files in seconds")
    azure_api_key = st.text_input("Azure API Key", type="password") 
    st.divider()

    pdf = st.file_uploader("Upload your PDF document", type="pdf")

    # Creating a button for users submit their PDF 
    submit = st.button("Generate Summary")

    # Load JSON file
    with open('keys.json', 'r') as file:
        data = json.load(file)

# Access to API KEYS
    MODEL = data["model"]
    AZURE_ENDPOINT = data["AZURE_OPENAI_ENDPOINT"]
    #AZURE_API_KEY = data["AZURE_OPENAI_API_KEY"]
    API_VERSION = data["api_version"]

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT
    os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key

    if submit:
        with get_openai_callback() as cost:
            response = summarizer(pdf)
            st.subheader("The cost is:")
            st.write(cost)

        st.subheader("Summary of file:")
        st.write(response)





# Python script execution starts here
if __name__ == '__main__':
    main() # Calling the main function