from indexing import get_pinecone_index, get_embedding, get_llm
from reranker import rerank_results
import pickle

# Retriever from Pinecone
def query_pinecone(query, top_k=3, use_rerank=False, use_hyde=False):
    if use_hyde:
        search_text = hyde_query(query)
        # print(f"HyDE: {search_text[:80]}...")
    else:
        search_text = query

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
    
    max_pinecone_score = max([r["score"] for r in results]) if results else 0
    # print(f"max_pinecone_score:{max_pinecone_score}")
    if use_rerank:
        results = rerank_results(query, results)[:top_k]
    else:
        results = results[:top_k]
    return results

def hyde_query(query):
    prompt = f"""Write a short factual passage that answers this question in 2-3 sentences.
                 Question: {query}
              """
    llm = get_llm()
    response = llm.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Send retrieved context to OpenAI
def generate_answer(query, context, model = 'gpt-4o'):
    context_str = "\n\n---\n\n".join(context)

#  old prompt Faithfulness: 0.971
#     prompt = f"""Answer the question using only the context below. If the question cannot be answered from the context above, say:
# "I don't know; this information is not in the provided context."

#                 Context:
#                 {context_str}

#                 Question:
#                 {query}

#                 Answer:
#                 """

    prompt = f"""Answer the question using only the context below.

                Rules:
                - Every key statement must be supported by the context
                - If possible, implicitly base your answer on multiple context parts
                - Do not introduce external knowledge
                - Be concise and precise

                If the context is insufficient, respond:
                "Not found in context."

                Context:
                {context_str}

                Question:
                {query}

            Answer:"""
    
    llm = get_llm()
    response = llm.chat.completions.create(
                    model = model,
                    messages=[
                        {"role": "system", "content" : "You are a helpful assistant that answers questions based only on the given context."},
                        {"role": "user", "content" :prompt},
                    ], max_tokens=512
                )
    return response.choices[0].message.content

def load_bm25_index():
    with open("bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]

def bm25_search(query, top_k=10):
    bm25, bm25_chunks = load_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    scored = sorted(zip(scores, bm25_chunks), key=lambda x: x[0], reverse=True)[:top_k]
    return [
        {"id": None, "text": chunk, "score": round(float(score), 4)}
        for score, chunk in scored
        if score > 0
    ]

def query_pinecone_hybrid(query, top_k=5, use_rerank=True):
    vector_results = query_pinecone(query, top_k=10, use_rerank=False)
    keyword_results = bm25_search(query, top_k=10)
    seen_texts = set()
    combined = []
    for r in vector_results + keyword_results:
        if r["text"] not in seen_texts:
            seen_texts.add(r["text"])
            combined.append(r)

    if use_rerank:
        combined = rerank_results(query, combined)[:top_k]
    else:
        combined = combined[:top_k]

    return combined


def main(): 
    query = "what is RAG?"
    context = query_pinecone_hybrid(query, top_k=5, use_rerank=True)
    texts = [r["text"] for r in context]
    response = generate_answer(query, texts)
    print(response)

    query = "who is the president of US?"
    context = query_pinecone_hybrid(query, top_k=5, use_rerank=True)
    texts = [r["text"] for r in context]
    response = generate_answer(query, texts)
    print(response)

if __name__ == "__main__":
    main()
