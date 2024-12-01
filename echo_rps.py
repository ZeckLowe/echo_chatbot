import os
from langchain_openai import OpenAIEmbeddings
import hashlib
from pinecone import Pinecone
from datetime import date

# Sample Transcript
from transcript1 import sample_transcript

# Initialize API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Retrieve organization name and meeting title from firebase
organization_name = input("Input organization name: ")
meeting_title = input("Input meeting title: ")

# Pinecone Initialization
PC = Pinecone(api_key=PINECONE_API_KEY)
INDEX = PC.Index(organization_name)

# OpenAI Initialization
EMBEDDINGS = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)

# Chunking Method
def chunk_text(text, max_chunk_size=500):
    # Ensure each text ends with a newline to correctly split sentences
    if not text.endswith("\n"):
        text += "\n"

    # Split text into sentence
    sentences = text.split("\n")
    chunks = []
    current_chunk = ""

    # Iterate over sentence and assemble chunks
    for sentence in sentences:
        # Check if adding the current sentence exceeds the maximum chunk size
        if (len(current_chunk) + len(sentences) + 2 > max_chunk_size and current_chunk):
            # Add the current chunk to the list and start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = ""
        # Add the current sentence to the current chunk
        current_chunk += sentence.strip() + "\n"
    # Add any remaining text as the last chunk
    if (current_chunk):
        chunks.append(current_chunk.strip())

    return chunks # type: list[str]

# Generate Embeddings Method
def generate_embeddings(texts):
    """
    Generate embeddings for a list of text.
    """
    embedded = EMBEDDINGS.embed_documents(texts)

    print("Generating embeddings: Done!")
    return embedded

# Generate Short ID Method
def generate_short_id(content):
    """
    Generate a short ID based on the content using SHA-256 hash.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))

    print("Generating short id: Done!")
    return hash_obj.hexdigest()

# Combine Texts and Vectors Method
def combine_vector_and_text(texts, meeting_title, date, text_embeddings):
    """
    Process a list of texts along with their embeddings.
    """
    data_with_metadata = []

    for doc_text, embedding in zip(texts, text_embeddings):
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        if not isinstance(meeting_title, str):
            meeting_title = str(meeting_title)

        if not isinstance(date, str):
            date = str(date)

        text_id = generate_short_id(doc_text)
        data_item = {
            "id": text_id,
            "values": embedding,
            "metadata": {"text": doc_text, "title": meeting_title, "date": date},
        }

        data_with_metadata.append(data_item)

    print("Combining vector and text: Done!")
    return data_with_metadata

# Upsert Data Method
def upsert_data_to_pinecone(data_with_metadata, namespace):
    """
    Upsert data with metadata into a Pinecone index.
    """
    INDEX.upsert(vectors=data_with_metadata, namespace=namespace)
    print("Upserting vectors to Pinecone: Done!")

# Main Method
def Pinecone(texts, organization, meeting_title):
    date_now = str(date.today()) # INITIALIZATION FOR DATE (DYNAMIC) BASED ON STORING
    print(date_now)
    namespace = meeting_title

    chunked_text = chunk_text(text=texts)
    chunked_text_embeddings = generate_embeddings(texts=chunked_text)
    data_with_meta_data = combine_vector_and_text(texts=chunked_text, meeting_title=organization, date=date_now,  text_embeddings=chunked_text_embeddings)
    upsert_data_to_pinecone(data_with_metadata=data_with_meta_data, namespace_name=namespace)

Pinecone(sample_transcript, organization_name, meeting_title)