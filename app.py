import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# =======================
# Load Models (CPU friendly)
# =======================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# =======================
# Global chat history
# =======================
chat_history = []

# =======================
# Helper functions
# =======================
def read_pdfs(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        reader = PdfReader(file.name if hasattr(file, 'name') else file)
        for page in reader.pages:
            all_text += page.extract_text()
    return all_text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_context(question, chunks, index):
    q_emb = embedding_model.encode([question])
    D, I = index.search(np.array(q_emb), k=3)
    context = " ".join([chunks[i] for i in I[0]])
    return context

def generate_answer(context, question):
    prompt = f"""
Previous conversation:
{' '.join(chat_history)}

Context:
{context}

User: {question}
Assistant:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {answer}")
    return answer

# =======================
# Gradio Function
# =======================
def chat_with_pdf(uploaded_files, question):
    if uploaded_files is None or question.strip() == "":
        return "Please upload PDFs and enter a question."

    # Step 1: Read PDFs
    text = read_pdfs(uploaded_files)

    # Step 2: Chunk
    chunks = chunk_text(text)

    # Step 3: FAISS index
    index, _ = build_faiss_index(chunks)

    # Step 4: Retrieve context
    context = retrieve_context(question, chunks, index)

    # Step 5: Generate answer
    answer = generate_answer(context, question)
    return answer

# =======================
# Gradio UI
# =======================
with gr.Blocks() as demo:
    gr.Markdown("# AI PDF Chat (RAG) – Free & CPU Friendly")
    with gr.Row():
        pdfs = gr.File(label="Upload PDFs", file_types=['.pdf'], file_types_multiple=True)
    question_input = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    output = gr.Textbox(label="Answer", placeholder="AI will answer here...")
    submit_btn = gr.Button("Ask")

    submit_btn.click(fn=chat_with_pdf, inputs=[pdfs, question_input], outputs=[output])

demo.launch()
