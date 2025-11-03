LLMStudy
=======

LLMStudy is a starter project that replicates a simplified NotebookLM-style workflow: ingest a lecture PDF, generate short notes, generate study questions, and answer user questions using a retrieval-augmented generation (RAG) pipeline.

This repository provides:
- PDF ingestion and text splitting
- Embeddings via sentence-transformers
- A FAISS vector index for retrieval
- LLM wrappers: OpenAI (preferred) with a small HF fallback for summarization, question generation and answers
- A minimal CLI to run ingest / notes / QA

Quick setup
-----------

This project uses Python 3.9+. Create and activate a virtualenv, then install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment variables:

- `OPENAI_API_KEY` - if set, LLMStudy will use OpenAI for generation (chat completions). If not set, it will try a small Hugging Face model as fallback.

Example usage (CLI)
-------------------

1. Ingest a PDF and build an index:

```bash
python scripts/llmstudy_cli.py ingest --pdf path/to/lecture.pdf --index-path data/index.faiss --meta-path data/meta.pkl
```

2. Generate short notes:

```bash
python scripts/llmstudy_cli.py notes --pdf path/to/lecture.pdf
```

3. Generate study questions:

```bash
python scripts/llmstudy_cli.py questions --pdf path/to/lecture.pdf --n 10
```

4. Ask a question (uses the built index):

```bash
python scripts/llmstudy_cli.py answer --query "What is gradient descent?" --index-path data/index.faiss --meta-path data/meta.pkl
```

Notes
-----

- This is a starter implementation. Models will be downloaded the first time they are used.
- For production use, consider managed vector stores, batching embeddings, and better prompt engineering.

Files created
-------------

- `src/llmstudy` - core implementation
- `scripts/llmstudy_cli.py` - simple CLI
- `requirements.txt` - Python deps
- `tests/test_split.py` - small unit test

Enjoy and iterate!


*Project Author Information**
------------------------------
- **Name:** Mayank Agrawal  
- **University:** Indian Institute of Technology (IIT) Kanpur  
- **Department:** Civil Engineering  

This information is added to associate this repository with my internship application.


