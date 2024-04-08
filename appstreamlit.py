import asyncio
import chainlit as cl
import pinecone
from langchain_community.vectorstores import Pinecone
import re
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

# Initialization of Pinecone and downloading of embeddings
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

@cl.on_chat_start
async def start():
    print("Chat session started. Sending greeting message...")
    await cl.Message(content="Hello, I am here to assist you with your queries.").send()

@cl.on_message
async def on_message(message: cl.Message):
    print(f"Received message: {message.content}")  # Print received message for debugging
    
    # Inform the user that their request is being processed
    processing_msg = cl.Message(content="Your request is being processed, please wait...")
    await processing_msg.send()

    # Run the synchronous qa.invoke function in a background thread
    result = await asyncio.to_thread(qa.invoke, {"query": message.content})
    
    if 'result' in result and 'source_documents' in result:
        answer = result["result"].replace('\\n', ' ').replace('\n', ' ')
        answer = remove_repeated_sentences(answer)
        sources = result["source_documents"]
        cleaned_sources = [f"Source {i+1}: {str(source).strip()}" for i, source in enumerate(sources)]
        answer_message = f"Answer: {answer}\n\nSource Documents:\n" + "\n".join(cleaned_sources)
    else:
        answer_message = "No internal results found."
    
    print(f"Sending response: {answer_message}")  # Print the response for debugging
    # Send the actual response
    await cl.Message(content=answer_message).send()

def remove_repeated_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    unique_sentences = set()
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.add(sentence)
    return ' '.join(unique_sentences)

if __name__ == "__main__":
    cl.run()
