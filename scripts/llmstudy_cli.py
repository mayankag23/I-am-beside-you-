#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from llmstudy.ingest import load_pdf_text, split_text
from llmstudy.rag import RAGIndex
from llmstudy.llm import generate_notes, generate_questions, answer_question_with_context


def cmd_ingest(args):
    text = load_pdf_text(args.pdf)
    chunks = split_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    idx = RAGIndex()
    idx.build(chunks)
    os.makedirs(os.path.dirname(args.index_path) or ".", exist_ok=True)
    idx.save(args.index_path, args.meta_path)
    print(f"Index saved to {args.index_path}, metadata to {args.meta_path}")


def cmd_notes(args):
    text = load_pdf_text(args.pdf)
    notes = generate_notes(text, n_sentences=args.n)
    print("--- Notes ---")
    print(notes)


def cmd_questions(args):
    text = load_pdf_text(args.pdf)
    qs = generate_questions(text, n=args.n)
    print("--- Questions ---")
    print(qs)


def cmd_answer(args):
    idx = RAGIndex.load(args.index_path, args.meta_path)
    # retrieve
    top = idx.retrieve(args.query, k=args.k)
    context_chunks = [t for t, d in top]
    ans = answer_question_with_context(args.query, context_chunks)
    print("--- Answer ---")
    print(ans)


def main():
    p = argparse.ArgumentParser(description="LLMStudy CLI")
    sub = p.add_subparsers(dest="cmd")

    ingest_p = sub.add_parser("ingest")
    ingest_p.add_argument("--pdf", required=True)
    ingest_p.add_argument("--index-path", default="data/index.faiss")
    ingest_p.add_argument("--meta-path", default="data/meta.pkl")
    ingest_p.add_argument("--chunk-size", type=int, default=1000)
    ingest_p.add_argument("--overlap", type=int, default=200)
    ingest_p.set_defaults(func=cmd_ingest)

    notes_p = sub.add_parser("notes")
    notes_p.add_argument("--pdf", required=True)
    notes_p.add_argument("--n", type=int, default=12)
    notes_p.set_defaults(func=cmd_notes)

    q_p = sub.add_parser("questions")
    q_p.add_argument("--pdf", required=True)
    q_p.add_argument("--n", type=int, default=10)
    q_p.set_defaults(func=cmd_questions)

    a_p = sub.add_parser("answer")
    a_p.add_argument("--query", required=True)
    a_p.add_argument("--index-path", default="data/index.faiss")
    a_p.add_argument("--meta-path", default="data/meta.pkl")
    a_p.add_argument("--k", type=int, default=5)
    a_p.set_defaults(func=cmd_answer)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
