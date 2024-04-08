import os
from google.cloud import storage
from dotenv import load_dotenv
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
#from langchain.vectorstores import Pinecone
import pinecone
from langchain_community.vectorstores import Pinecone

# Function to download PDFs from Google Cloud Storage
def download_pdfs_from_gcs(bucket_name, destination_folder):
    """Download all PDF files from a GCS bucket to a local directory."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs()
    for blob in blobs:
        if blob.name.endswith('.pdf'):
            destination_path = os.path.join(destination_folder, blob.name)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            blob.download_to_filename(destination_path)
            print(f"Downloaded {blob.name} to {destination_path}")

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Specify the absolute path to your data folder
DATA_FOLDER = r'C:\Users\hassa\Desktop\SRH-CHATBOT-V3\data'

# Download PDFs from Google Cloud Storage
BUCKET_NAME = 'srh-documents-bucket'  # Update this with your actual bucket name
download_pdfs_from_gcs(BUCKET_NAME, DATA_FOLDER)

# Process downloaded PDFs
extracted_data = load_pdf(DATA_FOLDER)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Define Pinecone index name
index_name = "srh-heidelberg-docs"

# Create embeddings for each of the text chunks and store them in Pinecone
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
