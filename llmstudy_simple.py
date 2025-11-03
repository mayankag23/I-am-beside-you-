#!/usr/bin/env python3
"""
LLMStudy - Simple PDF Study Assistant
A lightweight implementation without complex dependencies
"""

import os
import gradio as gr
from typing import List, Tuple

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber"""
    try:
        import pdfplumber
        texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
        return "\n\n".join(texts)
    except ImportError:
        return "Error: pdfplumber not installed. Run: pip install pdfplumber"
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start <= 0:
            start = end
        if start >= text_len:
            break
    
    return chunks

def generate_summary_notes(text: str) -> str:
    """Generate summary notes using OpenAI"""
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Limit text length to avoid token limits
        if len(text) > 3000:
            text = text[:3000] + "..."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at creating concise study notes. Generate clear, well-organized bullet points that capture the key concepts and important details."},
                {"role": "user", "content": f"Create concise study notes from this text:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating notes: {str(e)}"

def generate_study_questions(text: str, num_questions: int = 10) -> str:
    """Generate study questions using OpenAI"""
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Limit text length
        if len(text) > 3000:
            text = text[:3000] + "..."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert educator. Generate diverse study questions that test understanding of key concepts, including short answer, conceptual, and application questions."},
                {"role": "user", "content": f"Generate {num_questions} study questions from this text:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating questions: {str(e)}"

def simple_search(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Simple keyword-based search"""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:top_k]]

def answer_question(question: str, chunks: List[str]) -> str:
    """Answer question using relevant chunks"""
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Find relevant chunks
        relevant_chunks = simple_search(question, chunks, top_k=3)
        
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question."
        
        context = "\n\n".join(relevant_chunks)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided context. If the answer isn't in the context, say so clearly."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error answering question: {str(e)}"

def process_pdf(pdf_file, user_question):
    """Main function to process PDF and generate outputs"""
    if pdf_file is None:
        return "Please upload a PDF file.", "", "Please upload a PDF first."
    
    try:
        # Extract text from PDF
        pdf_text = extract_pdf_text(pdf_file.name)
        
        if pdf_text.startswith("Error"):
            return pdf_text, "", pdf_text
        
        # Split into chunks for processing
        chunks = split_into_chunks(pdf_text, chunk_size=800, overlap=100)
        
        if not chunks:
            return "No text found in PDF.", "", "No text found in PDF."
        
        # Generate summary notes (use first few chunks)
        summary_text = "\n\n".join(chunks[:3])  # Use first 3 chunks for summary
        notes = generate_summary_notes(summary_text)
        
        # Generate study questions
        questions = generate_study_questions(summary_text)
        
        # Answer user question if provided
        if user_question and user_question.strip():
            answer = answer_question(user_question.strip(), chunks)
        else:
            answer = "Ask a question about the document to get an answer!"
        
        return notes, questions, answer
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        return error_msg, error_msg, error_msg

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="LLMStudy - PDF Study Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📚 LLMStudy - PDF Study Assistant")
        gr.Markdown("Upload a PDF to get study notes, questions, and ask questions about the content!")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    file_types=[".pdf"], 
                    label="📄 Upload PDF File"
                )
                question_input = gr.Textbox(
                    placeholder="Ask a question about the document...", 
                    label="❓ Your Question",
                    lines=2
                )
                submit_btn = gr.Button("🚀 Process PDF", variant="primary")
        
        with gr.Row():
            with gr.Column():
                notes_output = gr.Textbox(
                    label="📝 Study Notes", 
                    lines=12,
                    placeholder="Study notes will appear here..."
                )
            
            with gr.Column():
                questions_output = gr.Textbox(
                    label="❓ Study Questions", 
                    lines=12,
                    placeholder="Study questions will appear here..."
                )
        
        with gr.Row():
            answer_output = gr.Textbox(
                label="💡 Answer", 
                lines=6,
                placeholder="Answer to your question will appear here..."
            )
        
        submit_btn.click(
            fn=process_pdf,
            inputs=[pdf_input, question_input],
            outputs=[notes_output, questions_output, answer_output]
        )
        
        # Auto-process when file is uploaded
        pdf_input.change(
            fn=lambda file: process_pdf(file, ""),
            inputs=[pdf_input],
            outputs=[notes_output, questions_output, answer_output]
        )
    
    return app

if __name__ == "__main__":
    print("🚀 Starting LLMStudy...")
    
    # Check if required packages are available
    try:
        import pdfplumber
        import openai
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install pdfplumber openai gradio")
        exit(1)
    
    # Create and launch the app
    app = create_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )