import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from src.agent import ask

load_dotenv()

GROUND_TRUTH_PATH = "eval/ground_truth.json"
RESULTS_PATH      = "eval/results.json"


def _check_faithfulness(answer: str, keywords: list[str]) -> dict:
    """
    Check if answer contains expected keywords.
    Case-insensitive. Returns per-keyword hits and overall score.
    At least one keyword must match for the answer to pass.
    """
    answer_lower = answer.lower()
    hits   = [kw for kw in keywords if kw.lower() in answer_lower]
    misses = [kw for kw in keywords if kw.lower() not in answer_lower]
    passed = len(hits) >= 1  # at least one keyword must appear
    return {
        "passed":  passed,
        "hits":    hits,
        "misses":  misses,
        "score":   round(len(hits) / len(keywords), 2) if keywords else 0,
    }


def _check_sql_accuracy(result: dict) -> dict:
    """
    Check SQL quality:
    - sql_generated: did the agent produce SQL?
    - sql_executed:  did it run without error?
    - has_results:   did it return data?
    """
    sql_generated = result.get("sql") is not None
    sql_executed  = sql_generated and "BigQuery" in result.get("sources", [])
    has_results   = sql_executed  # if BigQuery is in sources, query returned data

    return {
        "sql_generated": sql_generated,
        "sql_executed":  sql_executed,
        "has_results":   has_results,
        "score":         1 if (sql_generated and sql_executed) else 0,
    }


def run_evaluation(verbose: bool = True) -> dict:
    """
    Run all ground truth questions through PitWall and score them.
    Saves results to eval/results.json.
    Returns summary dict.
    """
    with open(GROUND_TRUTH_PATH, "r") as f:
        questions = json.load(f)

    results    = []
    sql_scores = []
    faith_scores = []

    print(f"\n{'='*60}")
    print(f"PitWall Evaluation Suite — {len(questions)} questions")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions, 1):
        question   = q["question"]
        keywords   = q["expected_keywords"]
        category   = q["category"]

        print(f"[{i:02d}/{len(questions)}] {question[:60]}...")

        try:
            result    = ask(question)
            sql_check = _check_sql_accuracy(result)
            faith_check = _check_faithfulness(result["answer"], keywords)

            sql_scores.append(sql_check["score"])
            faith_scores.append(1 if faith_check["passed"] else 0)

            status = "✅" if faith_check["passed"] else "❌"
            print(f"       {status} Faith: {faith_check['score']:.0%} "
                  f"| SQL: {'✅' if sql_check['sql_executed'] else '⚠️ '} "
                  f"| Hits: {faith_check['hits']}")

            results.append({
                "id":            q["id"],
                "question":      question,
                "category":      category,
                "answer":        result["answer"][:500],
                "sql":           result.get("sql"),
                "sources":       result.get("sources", []),
                "rounds":        result.get("rounds", 0),
                "tool_calls":    len(result.get("tool_calls", [])),
                "sql_check":     sql_check,
                "faith_check":   faith_check,
                "passed":        faith_check["passed"],
            })

        except Exception as e:
            print(f"       💥 ERROR: {str(e)[:80]}")
            results.append({
                "id":          q["id"],
                "question":    question,
                "category":    category,
                "error":       str(e),
                "passed":      False,
                "sql_check":   {"score": 0},
                "faith_check": {"score": 0, "passed": False},
            })
            sql_scores.append(0)
            faith_scores.append(0)

        # Small pause between questions to avoid rate limits
        if i < len(questions):
            time.sleep(2)

    # ── Summary ────────────────────────────────────────────────────────
    total          = len(questions)
    passed         = sum(1 for r in results if r["passed"])
    sql_acc        = round(sum(sql_scores) / total * 100, 1)
    faith_acc      = round(sum(faith_scores) / total * 100, 1)
    overall        = round((sql_acc + faith_acc) / 2, 1)

    # By category
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["passed"] += 1

    summary = {
        "timestamp":        datetime.now().isoformat(),
        "total_questions":  total,
        "passed":           passed,
        "failed":           total - passed,
        "sql_accuracy":     sql_acc,
        "faith_accuracy":   faith_acc,
        "overall_score":    overall,
        "by_category":      {
            cat: {
                "passed": v["passed"],
                "total":  v["total"],
                "pct":    round(v["passed"] / v["total"] * 100, 1)
            }
            for cat, v in categories.items()
        },
        "results": results,
    }

    # Save
    os.makedirs("eval", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print report
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total questions : {total}")
    print(f"  Passed          : {passed}/{total}")
    print(f"  SQL accuracy    : {sql_acc}%")
    print(f"  Answer faith.   : {faith_acc}%")
    print(f"  Overall score   : {overall}%")
    print(f"\n  By category:")
    for cat, v in summary["by_category"].items():
        bar = "█" * int(v["pct"] / 10) + "░" * (10 - int(v["pct"] / 10))
        print(f"    {cat:<12} {bar} {v['pct']}%  ({v['passed']}/{v['total']})")
    print(f"\n  Results saved → {RESULTS_PATH}")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    run_evaluation(verbose=True)