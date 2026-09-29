#!/bin/bash
# ============================================================================
# 咪咕大赛一键部署脚本（在咪咕仝学实例终端运行）
# 用法: bash auto_setup.sh
# 前提: 已将 https://github.com/43aquarius/tmpdata 克隆到 /mnt/storage/work
#
# 全流程: 装依赖 -> ModelScope下载Qwen3-4B -> 转换SFT数据 ->
#          LoRA训练(约2小时) -> 合并模型 -> 官方脚本端到端验证
# 幂等性: 每阶段完成会写标记文件 .done_<step>，中断后重跑自动跳过
# 日志:   /mnt/storage/workspace/pipeline.log + train.log
# ============================================================================
set -o pipefail

WORK=/mnt/storage/work                       # repo clone 目录
WS=/mnt/storage/workspace                    # 工作区(模型/数据/产物)
REPO_DATA="$WORK/数字人综合情感陪伴对话模型"
PART="$WS/participant"                       # 官方推理框架(从repo复制)
BASE_MODEL="$WS/base_model"                  # Qwen3-4B-Instruct-2507
FINAL_MODEL="$WS/model"                      # 合并后最终模型
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
cd "$WS" || exit 1
mkdir -p "$WS"
LOG="$WS/pipeline.log"
exec >>"$LOG" 2>&1   # 全部输出写日志(终端看不到,用 tail -f pipeline.log 看)
echo "==== pipeline start: $(date) ===="

step() {  # step <name> <command...>
  local name="$1"; shift
  if [ -f "$WS/.done_$name" ]; then echo "[skip] $name already done"; return 0; fi
  echo "---- [step $name] $(date) ----"
  "$@"
  if [ $? -eq 0 ]; then touch "$WS/.done_$name"; else echo "!!!! step $name FAILED, see log above"; exit 1; fi
}

# ---------- 1. 依赖 ----------
if [ ! -f "$WS/.done_deps" ]; then
  echo "---- [step deps] $(date) ----"
  pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple peft modelscope sentencepiece protobuf accelerate 2>&1 | tail -3
  python3 -c "import peft, transformers; print('peft', peft.__version__, '| transformers', transformers.__version__)" && touch "$WS/.done_deps" || { echo "deps FAILED"; exit 1; }
fi

# ---------- 2. 基座模型下载(ModelScope国内源) ----------
if [ ! -f "$WS/.done_model" ]; then
  echo "---- [step model] $(date) ----"
  if [ -f "$BASE_MODEL/config.json" ] && [ -f "$BASE_MODEL/model-00002-of-00002.safetensors" -o -f "$BASE_MODEL/model.safetensors" ]; then
    echo "base model already present"
  else
    python3 - <<'PYEOF'
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='/mnt/storage/workspace/base_model')
print('downloaded to', p)
PYEOF
  fi
  ls -lh "$BASE_MODEL" | head -8
  python3 -c "import json; c=json.load(open('$BASE_MODEL/config.json')); print('model_type:', c['model_type'])" && touch "$WS/.done_model" || { echo "model download FAILED"; exit 1; }
fi

# ---------- 3. SFT数据转换 ----------
if [ ! -f "$WS/.done_data" ]; then
  echo "---- [step data] $(date) ----"
  cp -r "$REPO_DATA/participant" "$PART" 2>/dev/null || true
  cp "$REPO_DATA/test_inference_data.jsonl" "$WS/" 2>/dev/null || true
  cp "$REPO_DATA/数字人综合情感陪伴对话模型"/*.py "$WS/" 2>/dev/null || cp "$REPO_DATA"/*.py "$WS/"
  # repo根目录下的脚本(migu_scripts/)
  cp -r "$WORK/migu_scripts/." "$WS/" 2>/dev/null || true
  python3 "$WS/convert_to_sft.py" --data "$REPO_DATA/训练-验证-数据集" --out "$WS" && touch "$WS/.done_data" || { echo "data convert FAILED"; exit 1; }
fi

# ---------- 4. LoRA训练(前台跑完,建议在tmux/nohup中运行本脚本) ----------
if [ ! -f "$WS/.done_train" ]; then
  echo "---- [step train] $(date) ----"
  python3 "$WS/train_lora.py" \
    --model_path "$BASE_MODEL" \
    --train_file "$WS/train_sft.jsonl" \
    --val_file "$WS/val_sft.jsonl" \
    --out "$FINAL_MODEL" \
    --epochs 2 --batch_size 2 --grad_accum 8 --max_len 3072 \
    --lora_r 16 --lora_alpha 32 --lr 1e-4 --save_steps 400 \
    2>&1 | tee "$WS/train.log" | grep -E "loss|it/s|%\|" | tail -200
  if [ -f "$FINAL_MODEL/config.json" ]; then touch "$WS/.done_train"; else echo "train FAILED"; exit 1; fi
fi

# ---------- 5. 配置官方推理框架 + 端到端验证(transformers后端) ----------
if [ ! -f "$WS/.done_verify" ]; then
  echo "---- [step verify] $(date) ----"
  python3 - <<'PYEOF'
import json
from pathlib import Path
cfg_p = Path('/mnt/storage/workspace/participant/configs/inference_config.json')
cfg = json.loads(cfg_p.read_text(encoding='utf-8'))
cfg['backend'] = 'transformers'
cfg['model_path'] = '/mnt/storage/workspace/model'
cfg['hardware_label'] = 'RTX4090-24G-x1-migu'
cfg['generation'] = {"max_new_tokens": 1024, "do_sample": False, "temperature": 1.0, "top_p": 1.0}
cfg_p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
print('inference_config updated')
PYEOF
  cd "$WS" && bash "$PART/start.sh" "$WS/test_inference_data.jsonl" "$WS/result"
  if [ -f "$WS/result/submission.jsonl" ]; then touch "$WS/.done_verify"; else echo "verify FAILED"; exit 1; fi
fi

echo "==== ALL DONE $(date) ===="
echo "产物: 模型=$FINAL_MODEL  验证结果=$WS/result/"
echo "下一步: 检查 result/submission.jsonl 内容质量后, 上传大赛镜像"
