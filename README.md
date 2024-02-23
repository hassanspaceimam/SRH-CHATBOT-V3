# SRH-CHATBOT-V3

SRH-CHATBOT-V3 is a sophisticated chatbot system tailored for university environments, designed to deliver efficient and precise responses by utilizing cutting-edge technologies such as Meta Llama2 and Pinecone. Engineered for simplicity and scalability, it is ideally suited for diverse university-related applications, from streamlining administrative tasks and improving operational efficiency to facilitating interactive learning environments for students and faculty. This chatbot aims to transform communication within the university, ensuring that information and academic support are more accessible and effective for the entire university community.


## Getting Started

Follow these steps to get your copy of SRH-CHATBOT-V3 up and running on your local machine for development and testing purposes.

### Prerequisites

- **Anaconda:** An Anaconda distribution to manage Python versions and environments. If you don't have Anaconda installed, download it from [Anaconda's official website](https://www.anaconda.com/products/individual).
- **Pinecone Account:** A Pinecone account to manage your vectors and indexes. Sign up at [Pinecone.io](https://www.pinecone.io/) if you haven't already.
- **Google Cloud Storage Bucket:** A Google Cloud Storage bucket for storing your PDF documents. Make sure you have created a bucket in your Google Cloud account.
- **Bing Search API Key:** A Bing Search API key for integrating Bing Web Search capabilities. Follow the steps at [Create a Bing Search Service Resource](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource) to obtain your API key.

## Steps to run project 

Clone the repository

```bash
Project repo: https://github.com/hassanspaceimam/SRH-CHATBOT-V3.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n schatbot python=3.8 -y
```

```bash
conda activate schatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "your_pinecone_api_key_here"
PINECONE_API_ENV = "your_pinecone_environment_here"
BING_SEARCH_API_KEY = "your_bing_search_api_key_here"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-13b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML
```

```bash
# Make sure Pinecone index called 'srh-heidelberg-docs' with METRIC = cosine and DIMENSIONS = 768 is created beforehand 
# run the following command to convert pdf files in data folder into vectors and upload them to Pinecone
python store_index.py
```

```bash
# Alternatively, if you wish to load your PDF documents from a Google Cloud Storage bucket, update the BUCKET_NAME in the gcs_store_index.py script. Specify the absolute path to your data folder in the DATA_FOLDER variable within this script, and then run:
python gcs_store_index.py
```

```bash
# Finally run the following command to launch app and go to http://localhost:8080/ for Flask or http://localhost:8501/ for Streamlit
python app.py  # For Flask
streamlit run appstreamlit.py  # For Streamlit
```
### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone
- Streamlit
- Google Cloud Storage
- Bing Search API

http://localhost:8080/` or the port specified by Flask upon launching.

### Tech Stack

- Python: Core programming language
- LangChain: Library for building language model applications
- Flask: Web framework for building the backend
- Meta Llama2: AI model for natural language understanding and generation
- Pinecone: Vector database for similarity search
- Streamlit: Framework for building interactive web apps
- Google Cloud Storage: Cloud storage for pdf files
- Bing Search API: Web search capabilities

### Acknowledgments

- Special thanks to SRH Heidelberg for supporting this project.
