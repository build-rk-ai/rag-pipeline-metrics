import cohere

co = cohere.Client()

def rerank_results(query, docs, top_k=50):
    # reranks docs by query relevance using Cohere, falls back to original order on failure
    texts = [doc["text"] for doc in docs if doc.get("text")]

    if not texts:
        return []

    try:
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=min(top_k, len(texts)),
        )

        reranked = []
        for r in response.results:
            orig_doc = docs[r.index]
            reranked.append({
                "id": orig_doc.get("id", f"rerank_{r.index}"),
                "text": orig_doc["text"],
                "score": round(float(r.relevance_score) * 100, 2),
                "cohere_raw": float(r.relevance_score)
            })

        return reranked

    except Exception as e:
        print(f"Rerank failed: {e}")
        return [{**doc, "score": doc.get("score", 0)} for doc in docs[:top_k]]