#!/usr/bin/env python3
"""Convert competition dataset to SFT training format.

Usage (on the training instance):
  python3 convert_to_sft.py --data <path-to-训练-验证-数据集> --out <workspace-dir>

Produces {out}/train_sft.jsonl and {out}/val_sft.jsonl plus system_prompt.txt.

Design (aligned with official scoring):
- Input: system prompt + conversation history rendered as chat messages
- Output: single JSON object {response_text, emotion_label, user_profile, memory_refs:[]}
- The system prompt is IDENTICAL to participant/run_inference.py SYSTEM_PROMPT,
  so training distribution matches inference distribution exactly.
"""
import argparse
import json
from pathlib import Path

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
]
PERSONALITY = ["extroverted", "introverted", "open", "conservative", "high_conscientiousness", "casual", "agreeable", "assertive", "emotionally_stable", "sensitive"]
INTERESTS = ["study_exam", "programming_technology", "reading_writing", "film_animation", "music", "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design"]
STYLES = ["brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous", "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user"]

SYSTEM_PROMPT = """你正在参加数字情感陪伴比赛。根据截止到最新用户消息的对话历史，完成下一轮 Assistant 回复预测。

只输出一个 JSON object，不要输出 Markdown 或解释。object 必须且只能包含：
- response_text: 非空中文回复，应自然、贴合上下文和用户偏好；
- emotion_label: 最新用户消息的情绪，只能从 {emotions} 中选择；
- user_profile: object，包含 personality_traits、interests、style 三个字符串数组；
- memory_refs: 固定输出 []，因为测试集没有公开 memory bank。

画像标签只能使用以下枚举；没有充分证据时宁可输出空数组：
personality_traits: {personality}
interests: {interests}
style: {styles}
""".format(
    emotions=", ".join(EMOTIONS),
    personality=", ".join(sorted(PERSONALITY)),
    interests=", ".join(sorted(INTERESTS)),
    styles=", ".join(sorted(STYLES)),
)


def history_to_messages(turns, target_user_turn_id):
    """Render conversation history up to (including) target user turn."""
    msgs = []
    for t in turns:
        if t["turn_id"] > target_user_turn_id:
            break
        msgs.append({"role": t["role"], "content": t["content"]})
    return msgs


def target_to_json_str(target):
    """Compact canonical JSON for the assistant output."""
    obj = {
        "response_text": target["response_text"],
        "emotion_label": target["emotion_label"],
        "user_profile": {
            "personality_traits": target["user_profile"].get("personality_traits", []),
            "interests": target["user_profile"].get("interests", []),
            "style": target["user_profile"].get("style", []),
        },
        "memory_refs": [],
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def convert(data_dir: Path, out_dir: Path, split: str) -> int:
    src = data_dir / split / ("train_public.jsonl" if split == "train" else "val_public.jsonl")
    out_path = out_dir / f"{split}_sft.jsonl"
    n = 0
    with open(src, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for pp in row["prediction_points"]:
                msgs = history_to_messages(row["turns"], pp["target_user_turn_id"])
                if not msgs or msgs[-1]["role"] != "user":
                    continue
                sample = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *msgs,
                        {"role": "assistant", "content": target_to_json_str(pp["target"])},
                    ]
                }
                w.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
    print(f"{split}: {n} SFT samples -> {out_path}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to 训练-验证-数据集 directory")
    ap.add_argument("--out", required=True, help="workspace output directory")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "train" / "train_public.jsonl").exists():
        raise SystemExit(f"ERROR: not found {data_dir}/train/train_public.jsonl")

    convert(data_dir, out_dir, "train")
    convert(data_dir, out_dir, "val")
    (out_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    print("system prompt saved")


if __name__ == "__main__":
    main()
