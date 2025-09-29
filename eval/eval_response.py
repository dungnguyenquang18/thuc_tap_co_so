import json
import time
from typing import List, Dict, Any, Optional

import requests

try:
	import sacrebleu
except Exception as e:
	sacrebleu = None
	print("[WARN] sacrebleu not available:", e)

try:
	from bert_score import score as bert_score
except Exception as e:
	bert_score = None
	print("[WARN] bert-score not available:", e)


DATASET_PATH = "d:/3Y2S/ttcs2/thuc_tap_co_so/make_data/data/test_data.json"
DETAILS_PATH = "d:/3Y2S/ttcs2/thuc_tap_co_so/eval/response_eval_details.json"
SUMMARY_PATH = "d:/3Y2S/ttcs2/thuc_tap_co_so/eval/response_eval_summary.json"


def load_json(path: str):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def call_server(query: str,
				base_url: str = "http://127.0.0.1:5001",
				path: str = "/api/chatbot",
				timeout: float = 60.0) -> Dict[str, Any]:
	url = base_url.rstrip("/") + path
	t0 = time.perf_counter()
	try:
		resp = requests.post(url, json={"query": query}, timeout=timeout)
		latency_ms = (time.perf_counter() - t0) * 1000
		if resp.status_code == 200:
			payload = resp.json()
			return {"ok": True, "answer": payload.get("answer", ""), "latency_ms": latency_ms}
		else:
			return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text}", "latency_ms": latency_ms}
	except Exception as e:
		latency_ms = (time.perf_counter() - t0) * 1000
		return {"ok": False, "error": str(e), "latency_ms": latency_ms}


def sentence_bleu(hyp: str, ref: str) -> Optional[float]:
	if sacrebleu is None:
		return None
	try:
		return float(sacrebleu.sentence_bleu(hyp, [ref]).score)
	except Exception:
		return None


def corpus_bleu(hyps: List[str], refs: List[str]) -> Optional[float]:
	if sacrebleu is None or not hyps:
		return None
	try:
		return float(sacrebleu.corpus_bleu(hyps, [refs]).score)
	except Exception:
		return None


def sentences_bert_f1(cands: List[str], refs: List[str], lang: str = "vi") -> Optional[List[float]]:
	if bert_score is None or not cands:
		return None
	try:
		P, R, F1 = bert_score(cands, refs, lang=lang)
		return [float(f) for f in F1.tolist()]
	except Exception:
		return None


def evaluate_server(dataset_path: str = DATASET_PATH,
					base_url: str = "http://127.0.0.1:5001",
					path: str = "/api/chatbot",
					timeout: float = 60.0,
					sleep_sec: float = 0.0,
					limit: Optional[int] = None) -> Dict[str, Any]:
	data = load_json(dataset_path)
	if limit is not None:
		data = data[:limit]

	details: List[Dict[str, Any]] = []
	hyps: List[str] = []
	refs: List[str] = []

	ok_count = 0

	for idx, item in enumerate(data[:], start=1):
		query = item.get("input") or item.get("question")
		expected = item.get("expect_response", "")

		print(f"{idx:03d}. Gọi server...", end=" ")
		result = call_server(query, base_url=base_url, path=path, timeout=timeout)

		if result.get("ok"):
			answer = result.get("answer", "")
			ok_count += 1
			bleu = sentence_bleu(answer, expected)
			details.append({
				"index": idx,
				"query": query,
				"expected": expected,
				"answer": answer,
				"latency_ms": result.get("latency_ms"),
				"bleu": bleu,
				"error": None,
			})
			hyps.append(answer)
			refs.append(expected)
			print(f"OK ({result.get('latency_ms'):.0f} ms)")
		else:
			# details.append({
			# 	"index": idx,
			# 	"query": query,
			# 	"expected": expected,
			# 	"answer": None,
			# 	"latency_ms": result.get("latency_ms"),
			# 	"bleu": None,
			# 	"error": result.get("error"),
			# })
			print("FAIL")

		if sleep_sec > 0:
			time.sleep(sleep_sec)

	# Metrics
	corpus_bleu_score = corpus_bleu(hyps, refs)
	bert_f1_list = sentences_bert_f1(hyps, refs, lang="vi")
	avg_sent_bleu = None
	if sacrebleu is not None:
		per_bleus = [d.get("bleu") for d in details if d.get("bleu") is not None]
		if per_bleus:
			avg_sent_bleu = float(sum(per_bleus) / len(per_bleus))

	avg_bert_f1 = None
	if bert_f1_list is not None and len(bert_f1_list) > 0:
		avg_bert_f1 = float(sum(bert_f1_list) / len(bert_f1_list))

	summary = {
		"total": len(data),
		"ok": ok_count,
		"coverage": (ok_count / len(data)) if data else 0.0,
		"corpus_bleu": corpus_bleu_score,
		"avg_sentence_bleu": avg_sent_bleu,
		"avg_bert_f1": avg_bert_f1,
	}

	# Attach BERT per-sentence if available
	if bert_f1_list is not None:
		j = 0
		for d in details:
			if d.get("answer") is not None:
				d["bert_f1"] = bert_f1_list[j]
				j += 1
			else:
				d["bert_f1"] = None

	# Save outputs
	with open(DETAILS_PATH, "w", encoding="utf-8") as f:
		json.dump(details, f, ensure_ascii=False, indent=2)
	with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	print("\nTóm tắt:")
	for k, v in summary.items():
		if isinstance(v, float):
			print(f"- {k}: {v:.4f}")
		else:
			print(f"- {k}: {v}")

	print(f"\nĐã lưu chi tiết vào: {DETAILS_PATH}")
	print(f"Đã lưu tổng hợp vào: {SUMMARY_PATH}")
	return {"summary": summary, "details": details}


if __name__ == "__main__":
	# Mặc định đánh giá toàn bộ bộ dữ liệu lên server local (port 5001)
	# Bạn có thể chỉnh limit=50 để chạy nhanh trước khi chạy full.
	evaluate_server(
		dataset_path=DATASET_PATH,
		base_url="http://127.0.0.1:5001",
		path="/api/chatbot",
		timeout=120.0,
		sleep_sec=0.0,
		limit=None,
	)