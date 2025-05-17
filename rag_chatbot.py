import streamlit as st
st.set_page_config(page_title="Physics RAG Chatbot", page_icon="🧪")

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ==== CONFIGURATION ====
INDEX_FILE = 'physics_faiss.index'
META_FILE = 'physics_metadata.json'
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

LLM_MODEL_PATH = 'models\mistral-7b-instruct-v0.1.Q4_K_M.gguf'
N_CTX = 4096
N_GPU_LAYERS = 32
TOP_K = 3
MAX_TOKENS = 256
TEMPERATURE = 0.7


# ==== LOAD COMPONENTS ====
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, 'r') as f:
        papers = json.load(f)
    return index, papers

@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device='cuda')
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        # n_threads=8,
        verbose=True
    )
    return embedder, llm

index, papers = load_index_and_metadata()
embedder, llm = load_models()

#==== HELPER FUNCTIONS ====
def retrieve_documents(query, k=TOP_K):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(np.array(query_vec), k)
    results = []
    for i in indices[0]:
        paper = papers[i]
        results.append({
            "title": paper.get('title', 'Unknown Title'),
            "abstract": paper.get('abstract', 'No abstract available.'),
            "url": paper.get('url', '')
        })
    return results


def build_prompt(query, docs):
    context = "\n\n".join([f"Title: {d['title']}\nAbstract: {d['abstract']}" for d in docs])
    return f"""You are a physics tutor who explains concepts clearly and simply. Start with a basic explanation, then expand if needed using the context provided.

Context:
{context}

Question: {query}

Answer:"""

def generate_answer(prompt):
    output_text = ""
    for chunk in llm(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["</s>", "Question:"],
        stream=True
    ):
        token = chunk["choices"][0]["text"]
        output_text += token
        yield token

# ==== STREAMLIT UI ====
st.title("🔬 Physics RAG Chatbot")
st.markdown("Ask a physics question based on research paper summaries.")

user_query = st.text_input("🧑‍💻 Your question:", key="input")

if st.button("Ask"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant papers..."):
            docs = retrieve_documents(user_query)

        if not docs:
            st.error("No relevant documents found.")
        else:
            prompt = build_prompt(user_query, docs)
            st.markdown("### 🤖 Answer:")
            response_placeholder = st.empty()
            response_text = ""
            for token in generate_answer(prompt):
                response_text += token
                response_placeholder.markdown(response_text)

            st.markdown("### 📄 Referred Papers:")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Paper {i}: {doc['title']}"):
                    st.markdown(f"**Title:** {doc['title']}")
                    st.markdown(f"**Abstract:** {doc['abstract']}")
                    st.markdown(f"[🔗 View on arXiv]({doc['url']})", unsafe_allow_html=True)
