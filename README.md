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
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-13b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML
```

```bash
# Make sure Pincecone index called 'srh-heidelberg-docs' with METRIC = cosine and DIMENSIONS = 768 is created beforehand 
# run the following command to convert pdf files in data folder into vectors and upload them to Pincecone
python store_index.py
```

```bash
# Finally run the following command to launch app and go to http://localhost:8080/ 
python app.py
```
### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone
- Streamlit