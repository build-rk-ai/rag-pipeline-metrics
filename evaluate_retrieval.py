import math
from tqdm import tqdm
from retrieval import query_pinecone, query_pinecone_hybrid
from indexing import get_pinecone_index, get_embedding
from sample_data import eval_questions, ground_truths

K_VALUES = [1, 3, 5]


def find_expected_hash(ground_truth_text, top_k=50):
    # finds the Pinecone chunk ID that best matches the ground truth via word overlap
    index = get_pinecone_index()
    gt_embedding = get_embedding(ground_truth_text)
    response = index.query(vector=gt_embedding, top_k=top_k, include_metadata=True)

    best_id, best_ratio = None, 0
    gt_words = set(ground_truth_text.lower().split())

    for match in response["matches"]:
        chunk_text = match["metadata"].get("text", "")
        chunk_words = set(chunk_text.lower().split())
        ratio = len(gt_words & chunk_words) / len(gt_words) if gt_words else 0
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = match["id"]

    return best_id


def evaluate_query(query, top_k, expected_hash, use_rerank=False, use_hyde=False, use_hybrid=False):
    # runs retrieval and computes precision@k and recall@k for a single query
    if use_hybrid:
        results = query_pinecone_hybrid(query, top_k=top_k, use_rerank=use_rerank)
    else:
        results = query_pinecone(query, top_k=top_k, use_rerank=use_rerank, use_hyde=use_hyde)

    retrieved_ids = [r["id"] for r in results]
    metrics = {}
    for k in K_VALUES:
        relevant_found = int(expected_hash in retrieved_ids[:k])
        metrics[f"precision@{k}"] = relevant_found / k
        metrics[f"recall@{k}"] = relevant_found

    return metrics, results


def mrr_at_k(results, relevant_ids, k=10):
    # returns reciprocal rank of the first relevant result within top k
    for i, r in enumerate(results[:k]):
        if r["id"] in relevant_ids:
            return 1.0 / (i + 1)
    return 0


def ndcg_at_k(results, relevant_ids, k=5):
    # measures ranking quality with position-discounted gains, normalized to ideal order
    top_k_ids = [r["id"] for r in results[:k]]
    rel_scores = [1 if doc_id in relevant_ids else 0 for doc_id in top_k_ids]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rel_scores))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / idcg if idcg > 0 else 0


def run_evaluation(use_rerank=False, use_hyde=False, use_hybrid=False):
    # maps ground truths to chunk IDs, runs all queries, prints precision, recall, MRR, NDCG
    print("Mapping ground truths to chunk IDs...")
    eval_data = []
    for question, gt in tqdm(zip(eval_questions, ground_truths)):
        expected_hash = find_expected_hash(gt)
        if expected_hash:
            eval_data.append((question, expected_hash))

    print(f"\nEvaluating {len(eval_data)} queries...\n")

    all_metrics = {f"precision@{k}": [] for k in K_VALUES}
    all_metrics.update({f"recall@{k}": [] for k in K_VALUES})
    mrr_list, ndcg_list = [], []

    for query, expected_hash in eval_data:
        metrics, results = evaluate_query(
            query, 10,
            expected_hash,
            use_rerank=use_rerank,
            use_hyde=use_hyde,
            use_hybrid=use_hybrid
        )
        for key in metrics:
            all_metrics[key].append(metrics[key])
        relevant_ids = [expected_hash]
        mrr_list.append(mrr_at_k(results, relevant_ids))
        ndcg_list.append(ndcg_at_k(results, relevant_ids))

    print("=" * 50)
    for k in K_VALUES:
        p = sum(all_metrics[f"precision@{k}"]) / len(eval_data)
        r = sum(all_metrics[f"recall@{k}"]) / len(eval_data)
        print(f"Precision@{k}: {p:.2%}")
        print(f"Recall@{k}   : {r:.2%}")
        print("-" * 30)

    print("\nProduction metrics:")
    print(f"MRR@10:  {sum(mrr_list)/len(mrr_list):.3f}")
    print(f"NDCG@5:  {sum(ndcg_list)/len(ndcg_list):.3f}")
    return all_metrics


if __name__ == "__main__":
    print("Baseline:")
    # run_evaluation()

    print("With Reranker:")
    run_evaluation(use_rerank=True)

    # print("With HyDE:")
    # run_evaluation(use_rerank=True, use_hyde=True)

    # print("With Hybrid:")
    # run_evaluation(use_rerank=True, use_hybrid=True)