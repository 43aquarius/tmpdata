#!/usr/bin/env bash
set -euo pipefail

# 启动虚拟环境conda_env, 名称固定，不要修改
source activate conda_env

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/configs/inference_config.json"

# 服务默认配置；可直接修改，也可用同名环境变量覆盖。
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-}"
START_TIMEOUT="${VLLM_START_TIMEOUT:-600}"
PYTHON_CMD="${PYTHON_BIN:-python3}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_ARGS=(--tensor-parallel-size 2) # 例如：VLLM_ARGS=(--tensor-parallel-size 2)

if [[ $# -ne 2 ]]; then
  echo "用法: $0 TEST_FILE RESULT_DIR" >&2
  exit 2
fi

TEST_FILE="$1"
RESULT_DIR="$2"

[[ -f "$TEST_FILE" && "$TEST_FILE" == *.jsonl ]] || {
  echo "ERROR: TEST_FILE 必须是存在的 .jsonl 文件: $TEST_FILE" >&2
  exit 1
}
if [[ -e "$RESULT_DIR" && ! -d "$RESULT_DIR" ]]; then
  echo "ERROR: RESULT_DIR 已存在但不是目录: $RESULT_DIR" >&2
  exit 1
fi
command -v "$PYTHON_CMD" >/dev/null 2>&1 || {
  echo "ERROR: 找不到 Python 命令: $PYTHON_CMD" >&2
  exit 1
}

BACKEND="${EVALUATION_BACKEND:-}"
if [[ -z "$BACKEND" ]]; then
  BACKEND="$("$PYTHON_CMD" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("backend", "transformers"))' \
    "$CONFIG_FILE")"
fi

if [[ "$BACKEND" == "transformers" ]]; then
  echo "后端: transformers（直接加载本地模型）"
  export EVALUATION_BACKEND="transformers"
  exec "$PYTHON_CMD" "$SCRIPT_DIR/run_inference.py" "$TEST_FILE" "$RESULT_DIR"
fi

if [[ "$BACKEND" != "openai_compatible" ]]; then
  echo "ERROR: backend 必须为 transformers 或 openai_compatible: $BACKEND" >&2
  exit 1
fi

CONFIG_MODEL_PATH="$("$PYTHON_CMD" -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("model_path", ""))' \
  "$CONFIG_FILE")"
MODEL_DIR="${VLLM_MODEL_DIR:-${EVALUATION_MODEL_PATH:-$CONFIG_MODEL_PATH}}"
if [[ -n "$MODEL_DIR" && "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$SCRIPT_DIR/$MODEL_DIR"
fi
MODEL_BASENAME="$(basename -- "$MODEL_DIR")"
MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$MODEL_BASENAME}"

[[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]] || {
  echo "ERROR: vLLM 模型目录不存在或缺少 config.json: $MODEL_DIR" >&2
  exit 1
}
command -v "$VLLM_BIN" >/dev/null 2>&1 || {
  echo "ERROR: 找不到 vLLM 命令: $VLLM_BIN" >&2
  exit 1
}
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1 && PORT <= 65535)) || {
  echo "ERROR: VLLM_PORT 必须是 1 到 65535 之间的整数" >&2
  exit 1
}
[[ "$START_TIMEOUT" =~ ^[0-9]+$ ]] && ((START_TIMEOUT >= 1)) || {
  echo "ERROR: VLLM_START_TIMEOUT 必须为正整数" >&2
  exit 1
}

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

CLIENT_HOST="$HOST"
[[ "$CLIENT_HOST" == "0.0.0.0" || "$CLIENT_HOST" == "::" ]] && CLIENT_HOST="127.0.0.1"
BASE_URL="http://$CLIENT_HOST:$PORT/v1"
HEALTH_URL="http://$CLIENT_HOST:$PORT/health"
LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/competition-vllm.XXXXXX.log")"

COMMAND=(
  "$VLLM_BIN" serve "$MODEL_DIR"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$MODEL_NAME"
  "${VLLM_ARGS[@]}"
)
[[ -n "$API_KEY" ]] && COMMAND+=(--api-key "$API_KEY")

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "启动 vLLM: $MODEL_DIR -> $BASE_URL"
echo "vLLM 日志: $LOG_FILE"
"${COMMAND[@]}" >"$LOG_FILE" 2>&1 &
VLLM_PID=$!

READY=0
for ((second = 0; second < START_TIMEOUT; second++)); do
  kill -0 "$VLLM_PID" 2>/dev/null || break
  if "$PYTHON_CMD" -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
    "$HEALTH_URL" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if ((READY == 0)); then
  echo "ERROR: vLLM 未能在 ${START_TIMEOUT}s 内就绪" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "vLLM 已就绪，开始推理测试"
export EVALUATION_BACKEND="openai_compatible"
export EVALUATION_BASE_URL="$BASE_URL"
export EVALUATION_API_MODEL="$MODEL_NAME"
export EVALUATION_API_KEY="$API_KEY"
export EVALUATION_HARDWARE_LABEL="${EVALUATION_HARDWARE_LABEL:-$(hostname)}"

"$PYTHON_CMD" "$SCRIPT_DIR/run_inference.py" "$TEST_FILE" "$RESULT_DIR"

echo "测试完成: $RESULT_DIR"
echo "vLLM 日志: $LOG_FILE"
