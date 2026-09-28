# Industrial Robot Troubleshooting Assistant

AI/ML academic mini-project: a **document-grounded** RAG assistant for industrial robot troubleshooting using Hugging Face Transformers, Sentence Transformers, FAISS, and Streamlit.

> This system is a **knowledge assistant only**. It does **not** control robots, clear alarms, or modify robot configuration.

## Important: First-Time Startup

After starting the application:

1. Open the application in the browser.
2. Click "Rebuild Index".
3. Wait for the rebuild to complete.
4. Click "Rebuild Index" a second time.
5. Wait for completion.
6. The application is then ready to answer questions.

*Do not claim that this is required on every startup unless the existing application behavior confirms that.*

## 1. Project overview

Technicians often struggle to locate error codes, maintenance steps, and safety procedures inside large manuals. This application:

1. Ingests industrial robot PDF manuals
2. Chunks text with source/page metadata
3. Embeds chunks with Sentence Transformers
4. Indexes vectors in FAISS
5. Retrieves top-K relevant passages for a natural-language question
6. Generates an answer with a Hugging Face instruction-tuned LLM using a strict grounding prompt

## 2. Features

- PDF ingestion (PyMuPDF) with page + section metadata
- Configurable chunking and TOP_K retrieval
- Persistent FAISS index (load without recomputing embeddings)
- Streamlit chat UI with source/page citations
- Runtime multi-PDF upload and index update
- Hallucination control for unknown questions
- Evaluation question set + retrieval evaluation script
- Safety-oriented prompting (manufacturer procedures / LOTO reminders)

## 3. Architecture

```text
Robot PDFs → Extract → Clean → Chunk → Embeddings → FAISS
                                                      │
User question → Query embedding → Retrieve top-K ─────┘
                                                      │
                                         Prompt + HF LLM → Answer + Sources
                                                      │
                                               Streamlit UI
```

## 4. Technology stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` default on this machine (set `MODEL_NAME` to `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-3B-Instruct` when download/VRAM allow) |
| Fallback LLM | Configurable via `FALLBACK_MODEL_NAME` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | FAISS (`faiss-cpu`) |
| PDF | PyMuPDF |
| UI | Streamlit |

### Hardware note (this development machine)

- GPU: NVIDIA GeForce RTX 3050 Laptop (6 GB)
- RAM: ~16 GB
- CUDA available via PyTorch

Default config uses **4-bit quantization** so the 3B model fits in 6 GB VRAM. Change `MODEL_NAME` / `LOAD_IN_4BIT` in `.env` if needed.

## 5. Installation

```bash
cd industrial_robot_assistant
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Copy environment defaults if needed:

```bash
copy .env.example .env
```

## 6. Dataset / document preparation

Sample educational manuals (original content for this project, **not** copied manufacturer manuals) can be generated with:

```bash
python scripts/generate_sample_manuals.py
```

They are written to:

```text
data/manuals/
  robot_user_manual.pdf
  maintenance_manual.pdf
  troubleshooting_manual.pdf
  safety_manual.pdf
```

You may replace these with real manufacturer PDFs you are licensed to use. Do not invent manufacturer facts outside the documents.

## 7. Index creation

```bash
python scripts/build_index.py --force
```

Index artifacts:

```text
vector_store/
  index.faiss
  chunks_meta.json
  manifest.json
```

## 8. Running the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically http://localhost:8501).

## 9. Example questions

- What does error E-123 mean?
- How do I troubleshoot alarm A-305?
- Why is the robot servo not turning on?
- What causes a controller communication error?
- When should the robot be recalibrated?
- What should I check if the robot overheats?
- What is the lockout/tagout procedure?
- What is the maximum payload of Robot XYZ? *(unknown — should refuse)*

## 10. Evaluation

Generate questions and run retrieval evaluation:

```bash
python scripts/generate_questions.py
python scripts/evaluate_retrieval.py
```

Unit tests:

```bash
pytest -q
```

Only report metrics from actual runs (see `evaluation/retrieval_report.json` after evaluation).

## 11. Limitations

- Answers are limited to uploaded/indexed documentation quality
- Sample manuals are educational stand-ins, not official OEM docs
- Small LLMs may paraphrase awkwardly even when context is correct
- Scanned image-only PDFs need OCR (not implemented)
- Not a robot controller or safety-certified system

## 12. Future improvements

- Voice interface, multilingual support, OCR for scanned manuals
- Cross-encoder reranking, larger domain models, fine-tuning
- Knowledge graphs, image-based fault ID, telemetry / predictive maintenance

## Project structure

```text
industrial_robot_assistant/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── data/manuals/
├── vector_store/
├── src/
│   ├── config.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   ├── prompt.py
│   └── pipeline.py
├── scripts/
├── tests/
└── evaluation/
```

## Safety disclaimer

Always follow the official manufacturer documentation, qualified technician guidance, lockout/tagout, and site regulations. This academic assistant never replaces those controls.
