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
import base64


SRH_IMAGE_PATH = 'SRH_rounded.png'
st.set_page_config(page_icon='SRH.png',page_title="SrhGPT")


def inject_custom_css():
    with open(SRH_IMAGE_PATH, "rb") as image_file:
        base64_logo = base64.b64encode(image_file.read()).decode()
    st.markdown("""
        <style>
        [data-testid="stHorizontalBlock"]{
                align-items:center;
                
        }
        [data-testid="stHorizontalBlock"] > div {
                width:100px !important;
                flex: none !important;
        }      
        h1 {
            color: #DF4807; 
            width:600px ! important;
            font-size: 40px !important;
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


bing_api_key = '0c485337974f4e6384f22f19d7086c1b'


PINECONE_API_KEY = "6ca2ecba-e336-4f09-9a39-e1e3b67d5f9d"
PINECONE_API_ENV = "gcp-starter"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

embeddings = download_hugging_face_embeddings()

index_name = "srh-heidelberg-docs"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = CTransformers(model="model/llama-2-13b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 612, 'temperature': 0.5})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# st.title("PersonaLearn: Personalized learning for ADSA")
def display_title_and_image():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(SRH_IMAGE_PATH, width=70)
    with col2:
        st.title("PersonaLearn: Personalized learning for ADSA")
    


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
    with st.spinner('Fetching the answer...'):
        if search_type == "Internal":
            result = qa.invoke({"query": user_input})
            if 'result' in result and 'source_documents' in result:
                answer = result["result"].replace('\\n', ' ').replace('\n', ' ')
                answer = remove_repeated_sentences(answer)

                sources = result["source_documents"]
                cleaned_sources = []
                for i, source in enumerate(sources):
                    source_str = str(source)
                    # Remove specific unwanted characters
                    cleaned_source = source_str.replace('page_content=', '').replace("'", "").replace('\\n', ' ').replace('\n', ' ')
                    cleaned_source = re.sub(r'[-.]{2,}', '', cleaned_source)  # This line removes sequences of dashes or dots
                    cleaned_source = remove_repeated_sentences(cleaned_source)
                    cleaned_sources.append(f"Source {i+1}: {cleaned_source.strip()}")  # Ensure no additional characters
                
                # Combine the cleaned answer and sources into the final message without unwanted characters
                answer_message = f"Answer: {answer}\n\nSource Documents:\n" + "\n".join(cleaned_sources)
            else:
                answer_message = "No internal results found."
        else:
            bing_result = bing_search(user_input, bing_api_key)
            if bing_result != "No results found.":
                clean_bing_result = bing_result.replace('\\n', ' ').replace('\n', ' ')
                clean_bing_result = re.sub(r'[-.]{2,}', '', clean_bing_result)  # Remove sequences of dashes or dots
                clean_bing_result = remove_repeated_sentences(clean_bing_result)
                answer_message = f"Bing Search Result: {clean_bing_result}"
            else:
                answer_message = "No external results found."
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(answer_message)
        st.session_state['source'].append(search_type)


inject_custom_css()
display_title_and_image()
initialize_session_state()
display_chat_history()
