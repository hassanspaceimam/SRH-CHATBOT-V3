import streamlit as st
import requests
from functools import lru_cache
from streamlit_chat import message
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

st.set_page_config(page_icon='SRH.png',page_title="SrhGPT")

def inject_custom_css():
    st.markdown("""
        <style>
        .msg {
            background-color: #FFFFFF;
        }
        .chat{
            background:#FFFF;
        }
        h1 {
            color: #DF4807; 
        }
        .message {
            background-color: none !important;
        }
        .overlay {
            position: fixed;
            display: block;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 2;
            cursor: pointer;
        }
        </style>
        """, unsafe_allow_html=True)

@lru_cache(maxsize=100)
def bing_search(query, bing_api_key):
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": query, "count": 1}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        results = []
        for result in data.get("webPages", {}).get("value", []):
            title = result.get("name")
            link = result.get("url")
            snippet = result.get("snippet")
            results.append(f"{title} - {snippet} - {link}")
        return ' | '.join(results) if results else "No results found."
    else:
        return "Error performing Bing search."


bing_api_key = '672569df979442e8abcf74be937a27d5'


PINECONE_API_KEY = "6ca2ecba-e336-4f09-9a39-e1e3b67d5f9d"
PINECONE_API_ENV = "gcp-starter"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

embeddings = download_hugging_face_embeddings()

index_name = "srh-heidelberg-docs"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = CTransformers(model="model/llama-2-13b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.5})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

st.title("SRH Chat Bot")


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'source' not in st.session_state:
        st.session_state['source'] = []

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me anything", key='input')
            internal_search_button = st.form_submit_button(label='Internal Search')
            external_search_button = st.form_submit_button(label='External Search')

        if user_input:
            if internal_search_button:
                search_type = "Internal"
            elif external_search_button:
                search_type = "External"
            else:
                search_type = None

            if search_type:
                process_query(user_input, search_type)

    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
            message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

import re

def remove_repeated_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return ' '.join(unique_sentences)

def process_query(user_input, search_type):
    with st.spinner('Fetching the answer...'):  # Start of spinner logic
        if search_type == "Internal":
            # Internal search with RetrievalQA
            result = qa.invoke({"query": user_input})
            if 'result' in result and 'source_documents' in result:
                # Remove newline characters and extra slashes from the answer
                answer = result["result"].replace('\\n', ' ').replace('\n', ' ')
                answer = remove_repeated_sentences(answer)  # Remove repeated sentences

                # Format the source documents
                sources = result["source_documents"]
                cleaned_sources = []
                for i, source in enumerate(sources):
                    # Convert the source document to string if it's not already a string
                    source_str = str(source)
                    # Remove unwanted parts from the source string and newline characters
                    cleaned_source = source_str.replace('page_content=', '').replace("'", "").replace('\\n', ' ').replace('\n', ' ')
                    cleaned_source = remove_repeated_sentences(cleaned_source)  # Remove repeated sentences
                    cleaned_sources.append(f"Source {i+1}: {cleaned_source}")
                    
                # Combine the cleaned answer and sources into the final message
                answer_message = f"Answer: {answer}\n\nSource Documents:\n" + "\n".join(cleaned_sources)
            else:
                answer_message = "No internal results found."
        else:
            # External search with Bing
            bing_result = bing_search(user_input, bing_api_key)
            if bing_result != "No results found.":
                # Remove newline characters from the Bing search result if needed
                clean_bing_result = bing_result.replace('\\n', ' ').replace('\n', ' ')
                clean_bing_result = remove_repeated_sentences(clean_bing_result)  # Remove repeated sentences
                answer_message = f"Bing Search Result: {clean_bing_result}"
            else:
                answer_message = "No external results found."
        
        # Update session state with the processed message
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(answer_message)
        st.session_state['source'].append(search_type)





inject_custom_css()
initialize_session_state()
display_chat_history()
