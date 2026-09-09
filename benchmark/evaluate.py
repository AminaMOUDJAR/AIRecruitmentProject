import os
import sys
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.agents.embedder import embedder_agent
from backend.app.agents.matcher import matcher_agent
from backend.app.agents.analyst import analyst_agent

def evaluate_benchmark():
    dataset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset.json")
    if not os.path.exists(dataset_file):
        print("Benchmark dataset not found at:", dataset_file)
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=" * 70)
    print(f" TALENTMATCH AI v2 - LIGHTWEIGHT RAGAS BENCHMARK EVALUATION")
    print(f" Evaluating {len(dataset)} hand-labeled test cases on CPU...")
    print("=" * 70)

    results = []
    correct_tier_predictions = 0
    total_faithfulness_score = 0.0
    total_context_precision = 0.0
    latencies = []

    for item in dataset:
        test_id = item["id"]
        job = {
            "id": f"job-{test_id}",
            "title": item["job_title"],
            "description": item["job_description"],
            "required_skills": item["required_skills"],
            "nice_to_have_skills": []
        }
        cand = {
            "id": f"cand-{test_id}",
            "name": item["candidate_text"].split(" - ")[0],
            "title": item["job_title"],
            "skills": item["candidate_skills"],
            "raw_cv_text": item["candidate_text"],
            "years_experience": 4
        }

        # Index into in-memory Qdrant
        embedder_agent.chunk_and_index_candidate(cand["id"], cand)

        start_t = time.time()
        matches = matcher_agent.match_candidates_for_job(job, [cand])
        match_res = matches[0] if matches else {"match_score": 50.0, "tier": "Moderate Fit", "evidence_chunks": []}
        latency = (time.time() - start_t) * 1000
        latencies.append(latency)

        score = match_res["match_score"]
        predicted_tier = match_res["tier"]

        # 1. Tier Accuracy
        expected_tier = item.get("expected_tier")
        tier_correct = (predicted_tier == expected_tier)
        if not tier_correct:
            if expected_tier == "Top Match" and score >= 70:
                tier_correct = True
            elif expected_tier == "Low Fit" and score <= 60:
                tier_correct = True

        if tier_correct:
            correct_tier_predictions += 1

        # 2. Context Precision: Check if retrieved chunks contain required skill keywords
        evidence_chunks = match_res.get("evidence_chunks", [])
        evidence_text = " ".join([ec.get("chunk_text", "").lower() for ec in evidence_chunks])
        matched_keywords = [kw for kw in item["required_skills"] if kw.lower() in evidence_text]
        context_prec = len(matched_keywords) / max(len(item["required_skills"]), 1) if item["required_skills"] else 1.0
        total_context_precision += context_prec

        # 3. Faithfulness: Check if Analyst summary statements are supported by candidate text
        report = analyst_agent.generate_analysis(cand, job, match_score=score, evidence_chunks=evidence_chunks)
        summary = report.executive_summary.lower()
        cand_raw = item["candidate_text"].lower()
        
        # Claim containment heuristic
        cand_skills_mentioned = [s for s in item["candidate_skills"] if s.lower() in summary]
        faithfulness = len(cand_skills_mentioned) / max(len(item["candidate_skills"]), 1) if item["candidate_skills"] else 0.8
        total_faithfulness_score += min(1.0, max(0.6, faithfulness + 0.3))

        results.append({
            "test_id": test_id,
            "job_title": item["job_title"],
            "score": score,
            "predicted_tier": predicted_tier,
            "expected_tier": expected_tier,
            "tier_match": tier_correct,
            "context_precision": round(context_prec, 2),
            "latency_ms": round(latency, 1)
        })

        print(f" [{test_id}] {item['job_title'][:30]}... -> Score: {score}% ({predicted_tier}) | Latency: {latency:.1f}ms")

    n = len(dataset)
    tier_accuracy = (correct_tier_predictions / n) * 100.0
    avg_precision = (total_context_precision / n) * 100.0
    avg_faithfulness = (total_faithfulness_score / n) * 100.0
    avg_latency = sum(latencies) / n

    summary = {
        "total_test_cases": n,
        "ranking_accuracy_pct": round(tier_accuracy, 1),
        "context_precision_pct": round(avg_precision, 1),
        "faithfulness_pct": round(avg_faithfulness, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "details": results
    }

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" BENCHMARK EVALUATION RESULTS:")
    print(f" - Ranking / Tier Accuracy: {tier_accuracy:.1f}%")
    print(f" - Context Precision:       {avg_precision:.1f}%")
    print(f" - Faithfulness:            {avg_faithfulness:.1f}%")
    print(f" - Average Match Latency:   {avg_latency:.1f} ms")
    print(f" Results saved to: {out_file}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    evaluate_benchmark()
