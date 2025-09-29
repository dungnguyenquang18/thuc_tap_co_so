from thuc_tap_co_so.src.ai_model.retrieve import Retrieve
import json
from rapidfuzz import fuzz
from typing import List, Dict
import dotenv
import os

dotenv.load_dotenv()

KEY = [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")]

#load json
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    



def _best_partial_ratio(expected: str, retrieved_blob: str) -> int:
    if not expected or not retrieved_blob:
        return 0
    # Compare against whole blob and per-line; take the max
    scores = [fuzz.partial_ratio(expected.lower(), retrieved_blob.lower())]
    for line in retrieved_blob.splitlines():
        if line.strip():
            scores.append(fuzz.partial_ratio(expected.lower(), line.lower()))
    return max(scores)


def evaluate_hit_k(ks: List[int] = [1, 3], threshold: int = 70) -> Dict[int, float]:
    """Evaluate Hit@k by checking if expected snippet appears (fuzzy) in top-k retrieved info.

    - ks: list of k values to evaluate.
    - threshold: partial_ratio threshold to count as a hit.

    Returns a dict mapping k -> accuracy (0..1)
    """
    data = load_json("d:/3Y2S/ttcs2/thuc_tap_co_so/make_data/data/test_data.json")


    results = {}
    details_per_k = {}
    total = len(data)
    for k in ks:
        hits = 0
        per_k_details = []
        print(f"\nĐánh giá Hit@{k} (ngưỡng khớp mờ >= {threshold})")
        print("=" * 60)
        for idx, item in enumerate(data, start=1):
            try:
                retriever = Retrieve(KEY[idx%3])
                query = item.get("input") or item.get("question")
                expected = item.get("expect_retrieve", "")
                info = retriever.retrieve_infomation(query, k=k)
                score = _best_partial_ratio(expected, info)
                is_hit = score >= threshold
                hits += 1 if is_hit else 0
                print(f"{idx:03d}. Hit={is_hit} | score={score} | Q: {query}")
                per_k_details.append({
                    "index": idx,
                    "k": k,
                    "query": query,
                    "expected": expected,
                    "score": score,
                    "hit": is_hit,
                    "retrieved": info,
                })
            except Exception as e:
                print(f"{idx:03d}. Lỗi api")
                
        acc = hits / total if total else 0.0
        results[k] = acc
        print(f"\nKết quả Hit@{k}: {hits}/{total} = {acc:.2%}")
        details_per_k[str(k)] = per_k_details

    with open("d:/3Y2S/ttcs2/thuc_tap_co_so/eval/hit_k_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open("d:/3Y2S/ttcs2/thuc_tap_co_so/eval/hitt_k_details.json", "w", encoding="utf-8") as f:
        json.dump(details_per_k, f, ensure_ascii=False, indent=2)
    return results


if __name__ == "__main__":
    # Chạy demo in thông tin truy xuất cho k=5


    # Chạy đánh giá Hit@1 và Hit@3
    evaluate_hit_k([1, 3], threshold=70)
    
    