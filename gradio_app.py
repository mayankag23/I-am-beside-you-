import gradio as gr
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

# Set OpenAI key
os.environ["OPENAI_API_KEY"] = "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"

try:
    from llmstudy.ingest import load_pdf_text, split_text
    from llmstudy.llm import generate_notes, generate_questions, answer_question_with_context
    print("LLMStudy modules imported successfully!")
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback functions...")
    MODULES_AVAILABLE = False
    
    # Fallback functions
    def load_pdf_text(path):
        import pdfplumber
        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
        return "\n\n".join(texts)
    
    def split_text(text, chunk_size=1000, overlap=200):
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap
            if start < 0:
                start = 0
            if start >= text_len:
                break
        return chunks
    
    def generate_notes(context, n_sentences=8):
        from openai import OpenAI
        client = OpenAI()
        prompt = f"Generate concise lecture notes (bullet points). Keep it short (about {n_sentences} bullets) and focused on key concepts.\n\nContext:\n{context}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that produces concise study notes from lecture material."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    
    def generate_questions(context, n=8):
        from openai import OpenAI
        client = OpenAI()
        prompt = f"Generate study questions (mixture of short answer and conceptual) from the following lecture material. Number them 1..{n}.\n\nContext:\n{context}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that generates study questions from lecture material."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    
    def answer_question_with_context(question, context_chunks):
        from openai import OpenAI
        client = OpenAI()
        ctx = "\n\n---\n\n".join(context_chunks)
        prompt = f"Use the context from the document to answer the question. If the answer is not in the context, say you don't know. Context:\n{ctx}\n\nQuestion: {question}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers based only on provided context. If asked beyond the context, say you don't know. Provide short, precise answers and cite the context when helpful."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()

def simple_text_search(query, chunks, k=5):
    """Simple keyword-based text search"""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [(text, score) for text, score in scored_chunks[:k]]

def llm_study_interface(pdf_file, question):
    try:
        if pdf_file is None:
            return "Please upload a PDF file", "", ""
        
        # Extract text from PDF
        pdf_text = load_pdf_text(pdf_file.name)
        chunks = split_text(pdf_text, chunk_size=500, overlap=100)
        
        # Generate notes from first few chunks
        sample_text = ' '.join(chunks[:3])
        notes = generate_notes(sample_text, n_sentences=8)
        
        # Generate questions
        questions = generate_questions(sample_text, n=8)
        
        # Answer user question if provided
        if question.strip():
            # Use simple text search for retrieval
            top_results = simple_text_search(question, chunks, k=5)
            context_chunks = [text for text, score in top_results]
            answer = answer_question_with_context(question, context_chunks)
        else:
            answer = "Please ask a question about the document."
        
        return notes, questions, answer
        
    except Exception as e:
        return f"Error: {str(e)}", "", f"Error processing: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=llm_study_interface,
    inputs=[
        gr.File(file_types=[".pdf"], label="Upload PDF"),
        gr.Textbox(placeholder="Ask a question about the document", label="Question")
    ],
    outputs=[
        gr.Textbox(label="Generated Notes", lines=10),
        gr.Textbox(label="Study Questions", lines=10),
        gr.Textbox(label="Answer", lines=5)
    ],
    title="LLMStudy - PDF Study Assistant",
    description="Upload a PDF and get notes, questions, and answers!",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Starting LLMStudy Gradio interface...")
    iface.launch(share=True, debug=True)