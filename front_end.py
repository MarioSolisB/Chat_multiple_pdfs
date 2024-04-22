import streamlit as st
from back_end import *
import os
import json

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

        response = summarizer(pdf)

        st.subheader("Summary of file:")
        st.write(response)





# Python script execution starts here
if __name__ == '__main__':
    main() # Calling the main function