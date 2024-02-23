# SRH-CHATBOT-V3
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
