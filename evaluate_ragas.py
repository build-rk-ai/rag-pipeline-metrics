import asyncio
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from tqdm import tqdm

from sample_data import eval_questions, ground_truths
from retrieval import query_pinecone, generate_answer, query_pinecone_hybrid

MODEL = 'gpt-4o'

def eval_rag_query(query, top_k, use_rerank=False, use_hyde=False, use_hybrid=False):
    if use_hybrid:
        results = query_pinecone_hybrid(query, top_k, use_rerank=use_rerank)
    else:
        results = query_pinecone(query, top_k, use_rerank=use_rerank, use_hyde=use_hyde)

    contexts = [r["text"] for r in results]
    answer = generate_answer(query, contexts, model=MODEL)

    return {
        "user_input": query,
        "response": answer,
        "retrieved_contexts": contexts
    }

sem = asyncio.Semaphore(10) 

async def score_single(name, scorer, record):
    async with sem:
        try:
            if name == "answer_relevancy":
                result = await scorer.ascore(
                    user_input=record["user_input"],
                    response=record["response"]
                )
            elif name == "context_precision":
                result = await scorer.ascore(
                    user_input=record["user_input"],
                    retrieved_contexts=record["retrieved_contexts"],
                    reference=record.get("reference")
                )
            elif name == "faithfulness":
                result = await scorer.ascore(
                    user_input=record["user_input"],
                    response=record["response"],
                    retrieved_contexts=record["retrieved_contexts"]
                )
            elif name == "context_recall":
                reference = record.get("reference")
                if not reference:
                    print(f"[context_recall] Skipping — no reference for: {record['user_input'][:60]}")
                    return None
                result = await scorer.ascore(
                        user_input=record["user_input"],
                        retrieved_contexts=record["retrieved_contexts"],
                        reference=reference
                    )
                        
            return getattr(result, "value", result)
        except Exception as e:
            # Handle the 429 specifically if it still happens
            if "rate_limit_exceeded" in str(e).lower():
                print(f"[{name}] Rate limit hit. Retrying in 2 seconds...")
                await asyncio.sleep(2)
                return await score_single(name, scorer, record)
            print(f"[{name}] ascore failed: {e}")
            return None


async def run_ragas_evaluation(use_rerank=False, use_hyde=False, use_hybrid=False, top_k=5):
    client = AsyncOpenAI()
    llm = llm_factory(MODEL, client=client)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)

    metrics = {
        "context_precision": ContextPrecision(llm=llm),
        "context_recall":    ContextRecall(llm=llm),
        "faithfulness":      Faithfulness(llm=llm),
        "answer_relevancy":  AnswerRelevancy(llm=llm, embeddings=embeddings),
    }

    records = []
    for query, gt in zip(eval_questions[:5], ground_truths[:5]):
        r = eval_rag_query(query, top_k, use_rerank, use_hyde, use_hybrid)
        r["reference"] = gt
        records.append(r)

    print("Running evaluation...")
    all_scores = {name: [] for name in metrics}

    # Debug ...
    # for r in records:
    #     print(f"Q: {r['user_input'][:60]}")
    #     print(f"A: {r['response'][:120]}")
    #     print(f"Contexts: {len(r['retrieved_contexts'])} chunks")
    #     print()

    for record in tqdm(records, desc="Evaluating", unit="query"):
        tasks = {
            name: asyncio.create_task(score_single(name, scorer, record))
            for name, scorer in metrics.items()
        }
        for name, task in tasks.items():
            all_scores[name].append(await task)

    print("\n" + "=" * 30 + "\n   RAGAS Evaluation\n" + "=" * 30)
    for name, values in all_scores.items():
        valid = [v for v in values if v is not None]
        mean = sum(valid) / len(valid) if valid else float("nan")
        print(f"{name.replace('_', ' ').title():>20}: {mean:.3f}")

if __name__ == "__main__":
    asyncio.run(run_ragas_evaluation(use_rerank=True, use_hyde=False, use_hybrid=False, top_k=5))