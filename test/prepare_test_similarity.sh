# --- 分布式处理配置 ---
NUM_WORKERS=8
MERGE_OUTPUT=true

# --- 输入数据路径 ---
DATA_PATH="/data_config/rag_data_val.yaml"
IMAGE_ROOT="/oven/oven_images"

# --- 知识库和模型路径 ---
KB_PATH="/infoseek/wiki_100_dict_v4.json"
FAISS_INDEX_PATH="/infoseek/infoseek_faiss/"
QFORMER_CKPT_PATH="/reranker/reranker.pth"

# --- vLLM 服务配置（需与服务端一致）---
VLLM_BASE_URL="Your root"
VLLM_MODEL="/cache/Qwen3-VL-8B-Instruct"   # 对应服务端 --served-model-name
VLLM_API_KEY="" #IF you have

# --- 输出路径 ---
OUTPUT_PATH="Your Path"

# --- 处理参数 ---
TOP_K=20
TOP_SECTIONS=3
ALPHA_1=0.5
MAX_SAMPLES=10000 #留空表示全部
DEVICE="cuda"
RETRIEVER_MODEL="eva-clip"

echo "=== 启动：带句子级相似度的对比数据预处理 ==="
echo "Data path: $DATA_PATH"
echo "Image root: $IMAGE_ROOT"
echo "KB path: $KB_PATH"
echo "FAISS index: $FAISS_INDEX_PATH"
echo "QFormer ckpt: $QFORMER_CKPT_PATH"
echo "vLLM base URL: $VLLM_BASE_URL"
echo "vLLM model: $VLLM_MODEL"
echo "Output path: $OUTPUT_PATH"
echo "Params: top_k=$TOP_K, top_sections=$TOP_SECTIONS, alpha_1=$ALPHA_1"
echo "Max samples: ${MAX_SAMPLES:-全部}"
echo "Workers: $NUM_WORKERS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================================"

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_PATH")"

# vLLM 可用性检查
echo "Checking vLLM server availability..."
if curl -s "$VLLM_BASE_URL/models" > /tmp/vllm_models.json; then
  echo "vLLM server reachable. Models served:"
  python - <<'PY'
import json,sys
try:
    j=json.load(open("/tmp/vllm_models.json"))
    print([m.get("id") for m in j.get("data",[])])
except Exception as e:
    print("Failed to parse /models:", e, file=sys.stderr)
PY
else
  echo "Warning: cannot reach vLLM server at $VLLM_BASE_URL"
fi

# 启动并行 workers（每个进程内部根据 worker_id 自动选 GPU）
pids=()
for ((worker_id=0; worker_id<NUM_WORKERS; worker_id++)); do
  echo "Starting worker $worker_id..."

  CMD="python /eval/prepare_test_with_similarity_comparison.py \
    --data_path \"$DATA_PATH\" \
    --output_path \"$OUTPUT_PATH\" \
    --image_root \"$IMAGE_ROOT\" \
    --kb_path \"$KB_PATH\" \
    --faiss_index_path \"$FAISS_INDEX_PATH\" \
    --qformer_ckpt_path \"$QFORMER_CKPT_PATH\" \
    --top_k $TOP_K \
    --top_sections $TOP_SECTIONS \
    --alpha_1 $ALPHA_1 \
    --num_workers $NUM_WORKERS \
    --worker_id $worker_id \
    --device \"$DEVICE\" \
    --retriever_model \"$RETRIEVER_MODEL\" \
    --vllm_base_url \"$VLLM_BASE_URL\" \
    --vllm_model \"$VLLM_MODEL\" \
    --vllm_api_key \"$VLLM_API_KEY\""

  if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
  fi

  eval $CMD &
  pids+=($!)
  sleep 2
done

echo "All workers started. PIDs: ${pids[@]}"
echo "Waiting for all workers to complete..."

# 等待所有 worker 完成
for pid in "${pids[@]}"; do
  if ! wait $pid; then
    echo "Worker with PID $pid exited with error"
    exit 1
  else
    echo "Worker with PID $pid completed successfully"
  fi
done

echo "All workers completed."

# 合并 worker 输出
base_name="$(dirname "$OUTPUT_PATH")/$(basename "$OUTPUT_PATH" .jsonl)"
final_output="${base_name}.jsonl"

if [ "$MERGE_OUTPUT" = true ]; then
  echo "Merging worker outputs into: $final_output"
  rm -f "$final_output"

  worker_files=()
  for ((worker_id=0; worker_id<NUM_WORKERS; worker_id++)); do
    wf="${base_name}_worker_${worker_id}.jsonl"
    if [ -f "$wf" ]; then
      worker_files+=("$wf")
      echo "Found worker file: $wf"
    else
      echo "Warning: Worker file not found: $wf"
    fi
  done

  if [ ${#worker_files[@]} -gt 0 ]; then
    cat "${worker_files[@]}" > "$final_output"
    total_samples=$(wc -l < "$final_output")
    echo "Final output: $final_output"
    echo "Total samples: $total_samples"

    echo "Checking <reason>, <question>, and sentence_similarities in output..."
    if grep -q "<reason>" "$final_output"; then
      echo "✅ Reasoning (<reason>) found."
    else
      echo "❌ No <reason> found."
    fi
    if grep -q "<question>" "$final_output"; then
      echo "✅ Refined questions (<question>) found."
    else
      echo "❌ No refined <question> found."
    fi
    if grep -q "sentence_similarities" "$final_output"; then
      echo "✅ Sentence-level similarities included."
    else
      echo "❌ No sentence-level similarities found."
    fi

    echo "Cleaning up worker files..."
    for wf in "${worker_files[@]}"; do
      rm -f "$wf"
      echo "Removed: $wf"
    done

    file_size=$(du -h "$final_output" | cut -f1)
    echo "Final file size: $file_size"
    echo "✅ Completed."
  else
    echo "❌ Error: No worker files found. Check logs."
    exit 1
  fi
else
  echo "MERGE_OUTPUT=false, skipping merge."
fi
