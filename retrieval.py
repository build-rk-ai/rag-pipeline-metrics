from indexing import get_pinecone_index, get_embedding, get_llm
from reranker import rerank_results
import pickle

bm25, bm25_chunks = None, None


def query_pinecone(query, top_k=5, use_rerank=False, use_hyde=False):
    # retrieves top_k chunks from Pinecone, optionally using HyDE and/or reranking
    search_text = hyde_query(query) if use_hyde else query
    query_embedding = get_embedding(search_text)
    index = get_pinecone_index()
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    if not response.get("matches"):
        return []

    results = []
    for match in response["matches"]:
        text = match["metadata"]["text"]
        if text.strip():
            results.append({
                "id": match["id"],
                "text": text,
                "score": round(match["score"] * 100, 2)
            })

    if use_rerank:
        results = rerank_results(query, results)[:top_k]
    else:
        results = results[:top_k]
    return results


def hyde_query(query):
    # generates a hypothetical answer to embed instead of the raw query
    prompt = f"Write a short factual passage that answers this question in 2-3 sentences. Question: {query}"
    llm = get_llm()
    response = llm.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def generate_answer(query, context, model='gpt-4o'):
    # sends retrieved context and query to GPT-4o and returns the answer
    context_str = "\n\n---\n\n".join(context)
    prompt = f"""Answer the question using only the context below. If the question cannot be answered from the context above, say:
"I don't know; this information is not in the provided context."

Context:
{context_str}

Question:
{query}

Answer:
"""
    llm = get_llm()
    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the given context."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512
    )
    return response.choices[0].message.content


def load_bm25_index():
    # loads the BM25 index and chunks saved during indexing
    with open("bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def bm25_search(query, top_k=10):
    # keyword search using BM25, returns top_k chunks by term frequency score
    global bm25, bm25_chunks
    if bm25 is None:
        bm25, bm25_chunks = load_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)