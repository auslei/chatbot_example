import streamlit as st
import ollama
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import Options
from chromadb.types import Collection

MODEL = "llama3.2"
CONTEXT_SIZE = 8192
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

COLLECTION_NAME = "pdf_chatbot"
DISTANCE_THRESHOLD = 0.5


st.set_page_config(layout="wide")
st.title("Chatbot")


# initialise variables
def initialise_variables():
    if "loaded_pdf" not in st.session_state:
        st.session_state.loaded_pdf = ""
    if "loaded_pdf_content" not in st.session_state:
        st.session_state.loaded_pdf_content = ""
    if "db" not in st.session_state:
        st.session_state.db = chromadb.PersistentClient("./db")
    if "collection" not in st.session_state:
        st.session_state.collection = \
            st.session_state.db.get_or_create_collection(name=COLLECTION_NAME, 
                                                         metadata={"hnsw:space": "cosine"}
                                                        )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        st.session_state.chat_history.append({"role": "system", "content": "You are a helpful assistant."})

def read_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def collection_exists(client, collection_name):
    existing_collections = [collection.name for collection in client.list_collections()]
    return collection_name in existing_collections

initialise_variables()

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Sidebar
st.sidebar.title("Upload PDF")
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# PROCESS PDF
if pdf_file is not None:
    # only process the pdf file if it has changed
    print(pdf_file.name, st.session_state.loaded_pdf)
    if st.session_state.loaded_pdf != pdf_file.name:
        st.session_state.loaded_pdf = pdf_file.name

        with st.spinner("Loading PDF..."):
            st.session_state.loaded_pdf_content = read_pdf(pdf_file)
            st.info(f'{pdf_file.name} loaded', icon="ℹ️")

        with st.spinner("Splitting PDF into chunks..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) # split the text into chunks
            chunks = splitter.split_text(st.session_state.loaded_pdf_content)
            st.info(f'{len(chunks)} loaded', icon="ℹ️")

        with st.spinner("Creating embeddings and storing in database..."):
            collection = st.session_state.collection
            all_ids = collection.get(include=[])['ids']
            if len(all_ids) > 0:
                    collection.delete(ids=all_ids)
            for i, chunk in enumerate(chunks):
                # Add chunk as a document with a unique id
                st.session_state.collection.add(
                    documents=[chunk],
                    metadatas=[{"source": pdf_file.name, "chunk_id": i}],
                    ids=[str(i)]
                )

                st.session_state.chat_history.append({"role": "system", 
                                                      "content":  f"The user has just uploaded a PDF file: {pdf_file.name}. It has been processed and stored into database for retrieval."})
                
elif st.session_state.loaded_pdf != "": # if the user has uploaded a pdf but then deleted it
        st.session_state.loaded_pdf_content = ""
        st.session_state.loaded_pdf = ""
        st.session_state.history.append({"role": "system", "content": f"The user has deleted the PDF file. "})

# PROCESS CHAT
if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(prompt)
    

    # Search the database for information related to the user's prompt
    if st.session_state.loaded_pdf_content != "":
        
        if st.session_state.collection is not None:
            # Find the most similar chunks to the user's prompt
            results = st.session_state.collection.query(
                query_texts=[prompt],
                n_results=5,
            )

            # Filter results based on the distance threshold
            filtered_results = {
                "documents": [],
                "distances": []
            }

            # Iterate over the results and filter based on the distance threshold
            for i, distance in enumerate(results["distances"][0]):
                if distance <= DISTANCE_THRESHOLD:
                    filtered_results["documents"].append(results["documents"][0][i])
                    filtered_results["distances"].append(distance)

            # If there are any filtered results, concatenate them into a single string
            if filtered_results["documents"]:
                st.info(f'Retrieving results from PDF content', icon="ℹ️")
                context = "\n\n".join(filtered_results["documents"])
                # Add the context to the user's prompt
                sys_prompt = f"""
                    Context:
                    {context}

                    Please answer the user's question in a clear, conversational way, using only the information in the context above. 
                    Make sure the response is direct, informative, and avoids repeating unnecessary details.
                """

                st.session_state.chat_history.append({"role": "system", "content": sys_prompt}) 

    st.session_state.chat_history.append({"role": "user", "content": prompt})  

    options = Options(num_ctx=CONTEXT_SIZE)
    response = ollama.chat(model=MODEL, 
                           options=options,
                           messages=st.session_state.chat_history, 
                           stream=True)
   
    def response_generator():
        assistant_response = ''
        for chunk in response:
            assistant_response += chunk['message']['content']
            yield chunk['message']['content']
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.write_stream(response_generator())

