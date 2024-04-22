import streamlit as st


#######################################
# BACKEND




#######################################
# FRONTEND 
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.text_input ("Ask a question about your documentos:")

    with st.sidebar:
        st.header("Introduce your API Key")
        azure_api_key = st.text_input("Azure API Key", type="password")
        st.text_input("OpenAI API Key", type="password")
        st.divider()
        ####
        st.header("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
        st.spinner("Processing")



if __name__ == '__main__':
    main()