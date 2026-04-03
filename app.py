import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="AI Knowledge Assistant")

st.title("📄 Personal AI Knowledge Assistant")

st.sidebar.title("AI Knowledge Assistant")
st.sidebar.write("Upload notes and ask questions from them.")

# Clear chat
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

if uploaded_file:

    pdf = PdfReader(uploaded_file)

    text = ""
    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content

    # Chunking
    chunk_size = 400
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Create embeddings
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Chat input (Enter key)
    question = st.chat_input("Ask a question about the document")

    if question:

        with st.spinner("Searching document..."):

            q_embed = model.encode([question])
            _, I = index.search(np.array(q_embed), k=5)

        top_answers = []

        for i in I[0]:
            if i < len(chunks):

                cleaned = chunks[i]

                # remove noisy PDF artifacts
                cleaned = cleaned.replace("\n", " ")
                cleaned = cleaned.replace("Cryptography & Network Security", "")
                cleaned = cleaned.replace("Module", "")
                cleaned = cleaned.replace("Dr", "")
                cleaned = cleaned.replace("GITAM UNIVERSITY", "")

                sentences = cleaned.split(".")

                # keep only meaningful sentences
                sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

                # prioritize definition-like sentences
                for s in sentences:
                    if (
                        " is " in s.lower()
                        or " refers to " in s.lower()
                        or " defined as " in s.lower()
                    ):
                        top_answers.append(s)
                        break

                if len(top_answers) >= 3:
                    break

        if not top_answers:
            answer = "I found related content but could not extract a clean definition."
        else:
            answer = "\n\n".join(top_answers)

        # Save conversation
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])