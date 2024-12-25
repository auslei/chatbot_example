import streamlit as st
import ollama
import base64
from ollama import Options

#MOODEL = "llama3.2-vision"
CONTEXT_SIZE = 8192


def image_to_base64(uploaded_file):
    """
    Convert a Streamlit uploaded file to a Base64-encoded string.
    
    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        str: Base64-encoded string of the file content, or None if no file is uploaded.
    """
    if uploaded_file is not None:
        # Read the uploaded file as binary
        file_bytes = uploaded_file.read()
        # Encode the binary data to base64
        base64_string = base64.b64encode(file_bytes).decode('utf-8')
        return base64_string
    else:
        return None


st.set_page_config(layout="wide")
st.title("Chatbot")

with st.sidebar:
    model = st.selectbox("Select a model", ["llama3.2", "llama3.2-vision", "llava"], index=1)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    st.session_state.chat_history.append({"role": "system", "content": "You are a helpful assistant."})

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(prompt)

    user_message = {"role": "user", "content": prompt}
    
    if uploaded_image is not None:
        image_base64 = image_to_base64(uploaded_image)
        user_message["images"] = [image_base64]

    st.session_state.chat_history.append(user_message)
    options = Options(num_ctx=CONTEXT_SIZE)
    params = {"model": model, 
              "options": options,
              "messages" : st.session_state.chat_history, 
              "stream": True}
    
    response = ollama.chat(**params)

    def response_generator():
        assistant_response = ''
        for chunk in response:
            assistant_response += chunk['message']['content']
            yield chunk['message']['content']
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.write_stream(response_generator())
