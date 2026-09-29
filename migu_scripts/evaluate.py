#!/usr/bin/env python3
"""Local evaluation replicating the official scoring metrics.

Official auto-eval (70 pts):
  - Emotion classification Acc (25)
  - Response quality: BLEU-4, ROUGE-L, BERTScore combined (20)
  - User profile inference: per-dimension average accuracy (15)
  - Inference latency: avg end-to-end delay (10)

Usage:
  python evaluate.py --pred submission.jsonl --gold val_public.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path("/home/z/my-project/migu_data/数字人综合情感陪伴对话模型/训练-验证-数据集")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------- BLEU ----------------
def ngrams(s, n):
    return [tuple(s[i:i+n]) for i in range(len(s)-n+1)]


def bleu_n(hyp, ref, n):
    h = ngrams(hyp, n)
    r = ngrams(ref, n)
    if not h or not r:
        return 0.0
    from collections import Counter
    hc, rc = Counter(h), Counter(r)
    overlap = sum(min(c, rc[g]) for g, c in hc.items())
    return overlap / len(h)


def bleu4(hyp, ref):
    # token-level for Chinese: chars are fine-grained; use jieba if available
    h, r = list(hyp), list(ref)
    if len(h) == 0 or len(r) == 0:
        return 0.0
    import math
    ps = []
    for n in range(1, 5):
        p = bleu_n(h, r, n)
        if p == 0:
            return 0.0
        ps.append(p)
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r)/len(h))
    return bp * math.exp(sum(math.log(p) for p in ps) / 4)


# ---------------- ROUGE-L ----------------
def lcs(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0]*(n+1)
    for i in range(1, m+1):
        prev = 0
        for j in range(1, n+1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def rouge_l(hyp, ref):
    h, r = list(hyp), list(ref)
    if not h or not r:
        return 0.0
    l = lcs(h, r)
    prec = l / len(h)
    rec = l / len(r)
    if prec + rec == 0:
        return 0.0
    return 2*prec*rec / (prec+rec)


# ---------------- Profile metrics ----------------
def profile_metrics(preds, golds):
    """Per-dimension accuracy: exact set match per dimension, averaged over 3 dims."""
    dims = ["personality_traits", "interests", "style"]
    correct = {d: 0 for d in dims}
    total = 0
    for p, g in zip(preds, golds):
        total += 1
        pp = p.get("user_profile") or {}
        gp = g.get("user_profile") or {}
        for d in dims:
            pv = set(pp.get(d) or [])
            gv = set(gp.get(d) or [])
            if pv == gv:
                correct[d] += 1
    return {d: correct[d]/max(total, 1) for d in dims}, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="submission.jsonl (predictions)")
    ap.add_argument("--gold", default=str(DATA / "val" / "val_public.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="evaluate first N only (0=all)")
    args = ap.parse_args()

    preds = load_jsonl(args.pred)
    gold_rows = load_jsonl(args.gold)

    # flatten gold to prediction-point level keyed by (conversation_id, target_user_turn_id)
    gold_map = {}
    for row in gold_rows:
        for pp in row["prediction_points"]:
            key = f"val_{row['conversation_id'].split('_')[-1]}_{pp['target_user_turn_id']}"
            gold_map[row["conversation_id"] + f"_t{pp['target_user_turn_id']}"] = pp["target"]

    # match preds to golds by id if available, else positional
    matched = []
    for p in preds:
        pid = p.get("id", "")
        if pid in gold_map:
            matched.append((p, gold_map[pid]))
        else:
            # try test-style id: testX_convNNNNNN_tN -> need mapping fallback
            continue

    # fallback: if less than half matched by id, use positional matching with val PPs
    if len(matched) < len(preds) * 0.5:
        gold_flat = []
        for row in gold_rows:
            for pp in row["prediction_points"]:
                gold_flat.append((row["conversation_id"] + f"_t{pp['target_user_turn_id']}", pp["target"]))
        matched = [(p, g) for (_, g), p in zip(gold_flat, preds)]
        print(f"(positional matching used: {len(matched)} pairs)")

    if args.limit:
        matched = matched[:args.limit]

    n = len(matched)
    print(f"evaluating {n} pairs")

    # metrics
    emo_correct = 0
    bleus, rouges = [], []
    pred_emotions = Counter()
    bert_scores = []

    for p, g in matched:
        pe = p.get("emotion_label", "")
        ge = g["emotion_label"]
        pred_emotions[pe or "(empty)"] += 1
        if pe == ge:
            emo_correct += 1
        bleus.append(bleu4(p.get("response_text", ""), g["response_text"]))
        rouges.append(rouge_l(p.get("response_text", ""), g["response_text"]))

    emo_acc = emo_correct / max(n, 1)
    avg_bleu = sum(bleus) / max(n, 1)
    avg_rouge = sum(rouges) / max(n, 1)

    # BERTScore (optional, heavy)
    try:
        from bert_score import BERTScorer
        import torch
        scorer = BERTScorer(lang="zh", device="cuda" if torch.cuda.is_available() else "cpu", rescale_with_baseline=False)
        P, R, F = scorer.score(
            [p.get("response_text", "") for p, _ in matched],
            [g["response_text"] for _, g in matched],
        )
        avg_bert = F.mean().item()
        bert_ok = True
    except Exception as e:
        print(f"[BERTScore unavailable: {str(e)[:80]}]")
        avg_bert = None
        bert_ok = False

    prof_accs, _ = profile_metrics([p for p, _ in matched], [g for _, g in matched])

    print("\n===== OFFICIAL-STYLE METRICS =====")
    print(f"Emotion Acc (25pts): {emo_acc:.4f}")
    print(f"BLEU-4:  {avg_bleu:.4f}")
    print(f"ROUGE-L: {avg_rouge:.4f}")
    if bert_ok:
        print(f"BERTScore F1: {avg_bert:.4f}")
        resp_quality = (avg_bleu + avg_rouge + avg_bert) / 3
    else:
        resp_quality = (avg_bleu + avg_rouge) / 2
    print(f"Response quality (20pts): {resp_quality:.4f}")
    print(f"Profile accuracies: " + ", ".join(f"{d}={v:.4f}" for d, v in prof_accs.items()))
    prof_avg = sum(prof_accs.values()) / 3
    print(f"Profile avg (15pts): {prof_avg:.4f}")
    print(f"\nPredicted emotion distribution (top 8): {pred_emotions.most_common(8)}")


if __name__ == "__main__":
    main()
