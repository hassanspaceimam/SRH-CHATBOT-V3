import streamlit as st
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

# Pinecone API configurations (Hardcoded for testing purposes only)
PINECONE_API_KEY = "6ca2ecba-e336-4f09-9a39-e1e3b67d5f9d"
PINECONE_API_ENV = "gcp-starter"

# Initialize Pinecone with hardcoded API details
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Load the Pinecone index
index_name = "srh-heidelberg-docs"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Setup the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Language Model Configuration
llm = CTransformers(model="model/llama-2-13b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.5})

# RetrievalQA Configuration
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Streamlit UI
st.title("Question Answering System")

# User input
user_query = st.text_input("Enter your question:", "")

# Handle query
if st.button("Submit"):
    try:
        if user_query:
            # Process the query using the RetrievalQA chain with invoke instead of call
            result = qa.invoke({"query": user_query})
            
            # Extracting the answer and the source documents
            if 'result' in result and 'source_documents' in result:
                answer = result["result"]
                # Here we assume 'source_documents' is a list of strings
                sources = result["source_documents"]
                # Display the answer
                st.write("Answer:", answer)
                # Display source documents. If it's a list, you'll need to format it.
                st.write("Source Documents:")
                for i, source in enumerate(sources):
                    st.write(f"Source {i+1}:", source)
            else:
                st.error("The response structure does not contain 'result' or 'source_documents' keys.")
        else:
            st.error("Please enter a question.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
