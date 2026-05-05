# RAG Pipeline — From Baseline to Production-Ready

A retrieval-augmented generation pipeline built on the AWS Prescriptive Guidance 
document on generative AI lifecycles. The goal was not just to build RAG but to 
measure what actually improves it.

## What I built

- PDF ingestion and chunking with RecursiveCharacterTextSplitter
- OpenAI text-embedding-3-small for embeddings
- Pinecone as the vector store
- GPT-4o for answer generation
- Cohere rerank-english-v3.0 for reranking
- BM25 hybrid search
- Evaluation framework tracking Precision@1, Recall@5, MRR@10, NDCG@5

## Results

| Approach        | Precision@1 | Recall@5 | MRR@10 |
|-----------------|-------------|----------|--------|
| Baseline        | 60%         | 85%      | 0.596  |
| + Reranker      | 65%         | 90%      | 0.662  |
| + HyDE          | 65%         | 90%      | 0.662  |
| + Hybrid Search | 65%         | 90%      | 0.662  |

Reranker was the only change that moved the numbers. HyDE and hybrid search 
added no measurable benefit for this document type.

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/build-rk-ai/rag-pipeline
cd rag-pipeline
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
Set up API keys
Add your keys to ~/.zprofile so they're available in every terminal session:

mac: 
export OPENAI_API_KEY=your-key-here
export PINECONE_API_KEY=your-key-here
export COHERE_API_KEY=your-key-here

windows:
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-key-here", "User")
[System.Environment]::SetEnvironmentVariable("PINECONE_API_KEY", "your-key-here", "User")
[System.Environment]::SetEnvironmentVariable("COHERE_API_KEY", "your-key-here", "User")
or 
via GUI: Search "Environment Variables" in the Start menu → "Edit the system environment variables" → "Environment Variables" → add under "User variables".


```

**4. Download the PDF**

Download the AWS Prescriptive Guidance document on generative AI lifecycles 
and place it in the `data/` folder as `lifecycle_genai.pdf`.

**5. Index the document**
```bash
python indexing.py
```

**6. Run a query**
```python
from retrieval import query_pinecone_hybrid, generate_answer

query = "What is RAG?"
context = query_pinecone_hybrid(query, top_k=5, use_rerank=True)
texts = [r["text"] for r in context]
print(generate_answer(query, texts))
```

**7. Run evaluation**
```bash
python evaluate_retrieval.py
```
