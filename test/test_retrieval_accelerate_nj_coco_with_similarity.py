import os
import json
import argparse
import logging
import datetime
from typing import List, Dict, Any, Optional
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper
import torch.nn.functional as F

"""
Variant of test_retrieval_accelerate_nj_coco.py that injects token-level similarity_scores
into the model for similarity-aware RoPE / attention compression.

Assumptions (due to lack of explicit retrieval markers in COCO preprocessed data):
- Each sample may contain a list of sentence_similarities (already computed externally).
- We map sentence-level similarities onto the input token sequence before generation.
- Non-mapped tokens receive a sentinel value (-1.0) so the model ignores them during retrieval-only top-K logic.
- If markers like "Here is the retrieval document:" are present, we restrict mapping to that span; otherwise we map across the full textual portion.

You can refine the mapping by customizing `build_token_similarity`.
"""

logger = logging.getLogger(__name__)
SENTINEL = -1.0

# ------------------------- Dataset and Collate ------------------------- #
class PreprocessedVLDataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.samples: List[Dict[str, Any]] = []
        if not data_path.endswith(".jsonl"):
            raise ValueError(f"Expected a .jsonl file, got: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        image = None
        if "images" in ex and ex["images"]:
            img_path = ex["images"][0]
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
        return {
            "image": image,
            "problem": ex.get("query", ""),
            "solution": ex.get("response", ""),
            "original_question": ex.get("question", ex.get("original_question", "")),
            "sentence_similarities": ex.get("sentence_similarities", None),
            "data_split": ex.get("data_split", ex.get("split", None)),
        }


def collate_fn(batch: List[Dict[str, Any]]):
    return {
        "images": [b["image"] for b in batch],
        "problems": [b["problem"] for b in batch],
        "solutions": [b["solution"] for b in batch],
        "original_questions": [b["original_question"] for b in batch],
        "sentence_similarities": [b.get("sentence_similarities") for b in batch],
        "data_split": [b.get("data_split") for b in batch],
    }

# ------------------------- Similarity Mapping ------------------------- #

def build_token_similarity(text_ctx: str, input_ids: torch.Tensor, sentence_sims: Optional[List[float]]) -> torch.Tensor:
    """Construct token-level similarity scores.

    Strategy:
    1. Initialize all tokens to SENTINEL (-1.0) so non-retrieval tokens are ignored.
    2. If sentence_sims present:
       - Try to locate a retrieval section delimited by markers.
       - If found, map sentences proportionally within that span.
       - Else, map across the entire textual token span (excluding leading special tokens heuristically).
    3. Linear allocation: divide retrieval span tokens evenly among sentences; each sentence's tokens share its similarity value.
    """
    seq_len = input_ids.shape[0]
    scores = torch.full((seq_len,), SENTINEL, dtype=torch.float32)
    if not sentence_sims or not isinstance(sentence_sims, (list, tuple)):
        return scores

    # Heuristic markers (align with rope_preprocessed_nj.py)
    start_marker = "Here is the retrieval document:"  # optional
    end_marker_candidates = [",\nShort answer:", "Short answer:"]  # prefer ",\nShort answer:" if present
    start_idx = text_ctx.find(start_marker)
    end_idx = -1
    for cand in end_marker_candidates:
        pos = text_ctx.find(cand)
        if pos != -1 and pos > start_idx:
            end_idx = pos
            break

    # Approximate char->token ratio
    ratio = seq_len / max(len(text_ctx), 1)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Map only inside retrieval block
        token_start = int((start_idx + len(start_marker)) * ratio)
        token_end = int(end_idx * ratio)
    else:
        # Fallback: use entire sequence (excluding first 2 special tokens if present)
        token_start = 2 if seq_len > 4 else 0
        token_end = seq_len

    token_start = max(0, min(token_start, seq_len - 1))
    token_end = max(token_start + 1, min(token_end, seq_len))
    span_len = token_end - token_start
    if span_len <= 0:
        return scores

    n_sent = len(sentence_sims)
    if n_sent == 0:
        return scores

    # Even partition boundaries
    # Each sentence covers [b[i], b[i+1]) token indices
    boundaries = [token_start + int(i * span_len / n_sent) for i in range(n_sent + 1)]

    for si in range(n_sent):
        s_val = float(sentence_sims[si])
        # Optional affine if outside [0,1]
        if s_val < 0.0 or s_val > 1.0:
            s_val = 0.5 * (s_val + 1.0)  # shift from [-1,1] -> [0,1]
        s_val = max(0.0, min(1.0, s_val))
        st = boundaries[si]
        ed = boundaries[si + 1]
        if ed > st:
            scores[st:ed] = s_val

    return scores

# ------------------------- Utilities (token suppression reused) ------------------------- #
class SuppressTokensProcessor(LogitsProcessor):
    def __init__(self, banned_token_ids: List[int]):
        self.banned = sorted(set(int(t) for t in banned_token_ids if t is not None))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.banned:
            return scores
        scores[:, self.banned] = -float("inf")
        return scores


def detect_image_token_ids(tokenizer) -> List[int]:
    candidates = [
        "<image>", "<|image|>", "<image_token>", "<image_placeholder>", "<|image_pad|>", "<IMG>", "<Image>",
    ]
    found: List[int] = []
    for tok in candidates:
        try:
            enc = tokenizer(tok, add_special_tokens=False)
            if enc and enc.get("input_ids"):
                found.extend(enc["input_ids"])
        except Exception:
            continue
    return sorted(set(int(t) for t in found if t is not None))

# ------------------------- Generation helpers (simplified from base script) ------------------------- #

def _build_warpers_from_config(gen_cfg):
    warpers = []
    try:
        temp = getattr(gen_cfg, "temperature", None)
        if temp is not None and temp != 1.0:
            warpers.append(TemperatureLogitsWarper(float(temp)))
    except Exception:
        pass
    try:
        top_k = getattr(gen_cfg, "top_k", 0)
        if top_k and top_k > 0:
            warpers.append(TopKLogitsWarper(int(top_k)))
    except Exception:
        pass
    try:
        top_p = getattr(gen_cfg, "top_p", 1.0)
        if top_p is not None and float(top_p) < 1.0:
            warpers.append(TopPLogitsWarper(float(top_p)))
    except Exception:
        pass
    return LogitsProcessorList(warpers) if warpers else None


def _build_stopping_criteria_from_config(gen_cfg, input_length: int):
    try:
        from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxNewTokensCriteria
        max_new = getattr(gen_cfg, "max_new_tokens", None)
        if max_new is not None:
            try:
                crit = MaxNewTokensCriteria(max_new_tokens=int(max_new), start_length=int(input_length))
            except TypeError:
                crit = MaxNewTokensCriteria(max_new_tokens=int(max_new))
            return StoppingCriteriaList([crit])
    except Exception:
        pass
    try:
        from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
        max_new = getattr(gen_cfg, "max_new_tokens", None)
        if max_new is not None:
            max_length = int(input_length) + int(max_new)
        else:
            max_length = getattr(gen_cfg, "max_length", None) or (int(input_length) + 1024)
        crit = MaxLengthCriteria(max_length=int(max_length))
        return StoppingCriteriaList([crit])
    except Exception:
        return None

def _compute_ac_from_sims(s):
    import math as _math
    if not s or not isinstance(s, (list, tuple)):
        return 0.0, 0.0, 0.0, 1.0
    vals = []
    needs_affine = any((float(x) < 0.0 or float(x) > 1.0) for x in s)
    for x in s:
        xv = float(x)
        if needs_affine:
            xv = 0.5 * (xv + 1.0)
        xv = 0.0 if _math.isnan(xv) else max(0.0, min(1.0, xv))
        vals.append(xv)
    M = len(vals)
    if M == 0:
        return 0.0, 0.0, 0.0, 1.0
    A = float(sum(vals) / float(M))
    ssum = float(sum(vals))
    if ssum <= 1e-12:
        H = 0.0
        C = 0.0
    else:
        p = [v / ssum for v in vals]
        H = -sum(pi * _math.log(max(pi, 1e-12)) for pi in p)
        C = 1.0 - (H / max(_math.log(M + 1e-12), 1e-12))
        C = max(0.0, min(1.0, C))
    AC = float(A * C)
    gain = 1.0 - AC
    return A, C, AC, gain

# ------------------------- Dual-stream zero-diff decode with similarity on ctx ------------------------- #
@torch.no_grad()
def decode_vl(
    args,
    model,
    processor,
    chat_messages_ctx: List[List[Dict[str, Any]]],
    chat_messages_pri: List[List[Dict[str, Any]]],
    images: List[Image.Image],
    sentence_sims: Optional[List[float]] = None,
):
    assert len(chat_messages_ctx) == 1 and len(chat_messages_pri) == 1 and len(images) == 1

    accelerator: Accelerator = args.accelerator
    base_model = model.module if hasattr(model, "module") else model

    # Build ctx/pri texts
    text_ctx = [processor.apply_chat_template(chat_messages_ctx[0], tokenize=False, add_generation_prompt=True)]
    text_pri = [processor.apply_chat_template(chat_messages_pri[0], tokenize=False, add_generation_prompt=True)]

    # Left padding
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    inputs_ctx = processor(text=text_ctx, images=images, padding=True, padding_side="left", return_tensors="pt")
    inputs_pri = processor(text=text_pri, images=images, padding=True, padding_side="left", return_tensors="pt")
    inputs_ctx = {k: v.to(accelerator.device) for k, v in inputs_ctx.items()}
    inputs_pri = {k: v.to(accelerator.device) for k, v in inputs_pri.items()}

    input_ids_ctx = inputs_ctx.get("input_ids")
    attention_mask_ctx = inputs_ctx.get("attention_mask")
    if attention_mask_ctx is None:
        attention_mask_ctx = torch.ones_like(input_ids_ctx)
    input_ids_pri = inputs_pri.get("input_ids")
    attention_mask_pri = inputs_pri.get("attention_mask")
    if attention_mask_pri is None:
        attention_mask_pri = torch.ones_like(input_ids_pri)

    # Build token similarity for ctx only
    tok_scores = build_token_similarity(text_ctx[0], input_ids_ctx[0], sentence_sims)
    if tok_scores.shape[0] != input_ids_ctx.shape[1]:
        L = input_ids_ctx.shape[1]
        tok_scores = tok_scores[:L] if tok_scores.shape[0] > L else torch.cat([tok_scores, torch.full((L - tok_scores.shape[0],), SENTINEL, dtype=torch.float32, device=tok_scores.device)], dim=0)
    similarity_scores_ctx = tok_scores.unsqueeze(0).to(dtype=torch.float32, device=accelerator.device)

    # Generation config and processors/warpers
    gen_cfg = deepcopy(base_model.generation_config)
    temp = getattr(args, "temperature", 0.7)
    gen_cfg.temperature = float(temp) if (temp is not None and temp > 0) else None
    gen_cfg.max_new_tokens = args.decode_depth
    gen_cfg.do_sample = True
    try:
        prepared = base_model._prepare_generation_config(gen_cfg)
        gen_cfg = prepared[0] if isinstance(prepared, tuple) else prepared
    except Exception:
        pass

    processors = base_model._get_logits_processor(
        generation_config=gen_cfg,
        input_ids_seq_length=input_ids_ctx.shape[1],
        encoder_input_ids=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=None,
    )
    try:
        warpers = base_model._get_logits_warper(gen_cfg)
    except AttributeError:
        warpers = _build_warpers_from_config(gen_cfg)
    try:
        names = [type(p).__name__ for p in (processors or [])]
        if any(n.endswith("LogitsWarper") or n in {"TemperatureLogitsWarper","TopKLogitsWarper","TopPLogitsWarper"} for n in names):
            warpers = None
    except Exception:
        pass
    # Optional suppression of image-like tokens (parity with base script)
    if getattr(args, "suppress_image_token", False) and hasattr(processor, "tokenizer"):
        banned_ids = detect_image_token_ids(processor.tokenizer)
        if banned_ids:
            if processors is None:
                processors = LogitsProcessorList([SuppressTokensProcessor(banned_ids)])
            else:
                processors = LogitsProcessorList(list(processors) + [SuppressTokensProcessor(banned_ids)])

    stopping_criteria = _build_stopping_criteria_from_config(gen_cfg, input_ids_ctx.shape[1])

    # Prefill both streams; pass similarity for ctx only
    def prefill(inputs, attn_mask, sim_scores=None):
        seq_len = inputs["input_ids"].shape[1]
        cache_position = torch.arange(seq_len, device=accelerator.device, dtype=torch.long)
        prefill_inputs = base_model.prepare_inputs_for_generation(
            input_ids=inputs.get("input_ids"),
            attention_mask=attn_mask,
            cache_position=cache_position,
            pixel_values=inputs.get("pixel_values", None),
            pixel_values_videos=inputs.get("pixel_values_videos", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            video_grid_thw=inputs.get("video_grid_thw", None),
            similarity_scores=sim_scores,
            use_cache=True,
        )
        if isinstance(prefill_inputs, dict) and prefill_inputs.get("position_ids") is not None:
            prefill_inputs["position_ids"] = prefill_inputs["position_ids"].clone().contiguous()
        else:
            pos_ids = cache_position.unsqueeze(0).repeat(inputs["input_ids"].size(0), 1).contiguous()
            prefill_inputs["position_ids"] = pos_ids
        return prefill_inputs

    out_ctx = model(**prefill(inputs_ctx, attention_mask_ctx, similarity_scores_ctx))
    out_pri = model(**prefill(inputs_pri, attention_mask_pri, None))
    past_key_values_ctx = out_ctx.past_key_values
    past_key_values_pri = out_pri.past_key_values

    # Fusion settings (logits_add, poe, dsab)
    fusion_mode = getattr(args, "fusion_mode", "logits_add")
    fusion_alpha = float(getattr(args, "fusion_alpha", 0.7))
    poe_beta = float(getattr(args, "poe_beta", 1.0))
    dsab_stage = getattr(args, "dsab_stage", "pre_warpers")

    # DSAB stats
    vocab_size = None
    try:
        vocab_size = int(getattr(base_model.get_output_embeddings(), "out_features", None) or base_model.lm_head.out_features)
    except Exception:
        try:
            vocab_size = int(processor.tokenizer.vocab_size)
        except Exception:
            pass
    context_freq = None
    hist_counts = None
    gen_len = 0
    if fusion_mode == "dsab":
        with torch.no_grad():
            ctx_tokens = input_ids_ctx[0].detach().to("cpu")
            v = vocab_size if vocab_size is not None else int(out_ctx.logits.shape[-1])
            context_len = max(int(ctx_tokens.numel()), 1)
            freq = torch.bincount(ctx_tokens, minlength=v).float() / float(context_len)
            context_freq = freq.to(accelerator.device).unsqueeze(0).unsqueeze(1)
            hist_counts = torch.zeros((1, v), device=accelerator.device)

    # Sentence gain (1-AC)
    try:
        _, _, _, gain_val = _compute_ac_from_sims(sentence_sims)
    except Exception:
        gain_val = 0.0
    sentence_gain_t = torch.tensor([[[float(gain_val)]]], device=accelerator.device, dtype=torch.float32)

    generated_ids: List[int] = []

    while True:
        logits_ctx_step = out_ctx.logits[:, -1, :].float()
        logits_pri_step = out_pri.logits[:, -1, :].float()

        if fusion_mode == "poe":
            logits_step = torch.log_softmax(logits_ctx_step, dim=-1) + poe_beta * torch.log_softmax(logits_pri_step, dim=-1)
        elif fusion_mode == "dsab":
            # Strict DSAB-lite in pre_warpers (default) or strict mode similar to reference
            score1 = logits_ctx_step.unsqueeze(1)
            score2 = logits_pri_step.unsqueeze(1)
            if getattr(args, "dsab_lite", False) and dsab_stage == "pre_warpers":
                p_ctx0 = F.softmax(score1, dim=-1)
                p_pri0 = F.softmax(score2, dim=-1)
                log_p_ctx0 = F.log_softmax(score1, dim=-1)
                log_p_pri0 = F.log_softmax(score2, dim=-1)
                H_ctx0 = -(p_ctx0 * log_p_ctx0).sum(dim=-1, keepdim=True)
                H_pri0 = -(p_pri0 * log_p_pri0).sum(dim=-1, keepdim=True)
                Delta_H0 = H_pri0 - H_ctx0
                alpha_gamma = float(getattr(args, "dsab_alpha_gamma", 1.0))
                use_d2 = bool(getattr(args, "dsab_alpha_use_d2", False))
                if use_d2:
                    inner0 = torch.pow(p_ctx0.clamp(min=1e-12), 0.5)
                    D2_0 = 1.0 - (p_pri0 * inner0).sum(dim=-1, keepdim=True)
                    D2_norm0 = torch.clamp(D2_0 / (1.0 + D2_0), 0.0, 1.0)
                    raw_input = alpha_gamma * Delta_H0 + D2_norm0
                else:
                    raw_input = alpha_gamma * Delta_H0
                ac_gain_gamma = float(getattr(args, "ac_gain_gamma", 1.0))
                if ac_gain_gamma != 0.0:
                    raw_input = raw_input + ac_gain_gamma * sentence_gain_t
                a_raw = torch.sigmoid(raw_input)
                a_min = float(getattr(args, "dsab_alpha_min", 0.2))
                a_max = float(getattr(args, "dsab_alpha_max", 0.9))
                alpha_hat = torch.clamp(a_raw, a_min, a_max)
                mix_space = getattr(args, "dsab_mix_space", "logits")
                if mix_space == "logprob":
                    mix_ctx = F.log_softmax(logits_ctx_step, dim=-1)
                    mix_pri = F.log_softmax(logits_pri_step, dim=-1)
                else:
                    mix_ctx = logits_ctx_step
                    mix_pri = logits_pri_step
                logits_step = (alpha_hat.squeeze(1) * mix_ctx) + ((1.0 - alpha_hat.squeeze(1)) * mix_pri)
            else:
                # Fallback to simple logits_add if strict path not implemented here
                logits_step = fusion_alpha * logits_ctx_step + (1.0 - fusion_alpha) * logits_pri_step
        else:
            logits_step = fusion_alpha * logits_ctx_step + (1.0 - fusion_alpha) * logits_pri_step

        if processors is not None:
            logits_step = processors(input_ids_ctx, logits_step)
        if warpers is not None:
            logits_step = warpers(input_ids_ctx, logits_step)
        probs = torch.softmax(logits_step, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        input_ids_ctx = torch.cat([input_ids_ctx, next_token.to(input_ids_ctx.device)], dim=1)
        attention_mask_ctx = torch.cat([attention_mask_ctx, torch.ones_like(next_token)], dim=1)
        input_ids_pri = torch.cat([input_ids_pri, next_token.to(input_ids_pri.device)], dim=1)
        attention_mask_pri = torch.cat([attention_mask_pri, torch.ones_like(next_token)], dim=1)
        generated_ids.append(next_token.item())

        eos_token_id = base_model.generation_config.eos_token_id
        eos_list = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        if next_token.item() in eos_list:
            break
        if stopping_criteria is not None and stopping_criteria(input_ids_ctx, None):
            break
        if len(generated_ids) >= args.decode_depth:
            break

        def step_with_prepare(inputs_ids, attn_mask, pkv):
            last_ids = inputs_ids[:, -1:]
            cache_pos = (attn_mask.sum(dim=-1) - 1).to(dtype=torch.long)
            prepared = base_model.prepare_inputs_for_generation(
                input_ids=last_ids,
                attention_mask=attn_mask,
                past_key_values=pkv,
                cache_position=cache_pos,
                pixel_values=None,
                similarity_scores=None,
                use_cache=True,
            )
            if isinstance(prepared, dict):
                if prepared.get("position_ids") is not None:
                    prepared["position_ids"] = prepared["position_ids"].clone().contiguous()
                else:
                    prepared["position_ids"] = cache_pos.view(-1, 1).contiguous()
            return prepared

        out_ctx = model(**step_with_prepare(input_ids_ctx, attention_mask_ctx, past_key_values_ctx))
        out_pri = model(**step_with_prepare(input_ids_pri, attention_mask_pri, past_key_values_pri))
        past_key_values_ctx = out_ctx.past_key_values
        past_key_values_pri = out_pri.past_key_values

    decoded = processor.batch_decode([torch.tensor(generated_ids, device=accelerator.device)],
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
    return decoded

# ------------------------- Evaluation Loop ------------------------- #

def evaluate(args):
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=259200))])
    args.accelerator = accelerator

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    dataset = PreprocessedVLDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = accelerator.prepare(model)

    all_outputs = []
    all_questions = []
    all_solutions = []
    all_original_questions = []
    all_data_splits = []
    all_sentence_similarities = []

    from tqdm import tqdm
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_main_process, desc="decode+sim")

    for batch in dataloader:
        for i in range(len(batch["problems"])):
            img = batch["images"][i]
            if img is not None:
                w, h = img.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28 / max(w, 1)))
                    else:
                        new_h = 28
                        new_w = int(w * (28 / max(h, 1)))
                    try:
                        resample = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample = Image.LANCZOS
                    img = img.resize((new_w, new_h), resample=resample)

            full_query = (batch["problems"][i] or "").replace("<image>", "").strip()
            orig_q = (batch["original_questions"][i] or "").replace("<image>", "").strip()
            sims = batch.get("sentence_similarities", [None])[i]

            msg_ctx = [[
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_query},
                ]}
            ]]
            msg_pri = [[
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{orig_q}\nShort answer:"},
                ]}
            ]]

            try:
                gen = decode_vl(args, model, processor, msg_ctx, msg_pri, [img], sentence_sims=sims)
            except Exception as e:
                err_msg = str(e)
                rope_oob = ("get_rope_index" in err_msg) or ("out of bounds" in err_msg) or ("position_ids" in err_msg)
                if rope_oob:
                    if accelerator.is_main_process:
                        logger.warning(f"[skip] ROPE index error on sample {i}: {err_msg} — skipping.")
                    continue
                else:
                    raise

            if accelerator.is_main_process:
                all_outputs.append(gen)
                all_questions.append(batch["problems"][i])
                all_solutions.append(batch["solutions"][i])
                all_original_questions.append(batch["original_questions"][i])
                all_data_splits.append(batch.get("data_split", [None])[i])
                all_sentence_similarities.append(sims)

        if accelerator.is_main_process:
            pbar.update(1)

    if accelerator.is_main_process:
        pbar.close()
        out_path = f"{args.output_path}{accelerator.process_index}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {"rank": accelerator.process_index, "num_samples": len(all_outputs)},
                "question": all_original_questions,
                "full_query": all_questions,
                "results": all_outputs,
                "solution": all_solutions,
                "data_split": all_data_splits,
                "sentence_similarities": all_sentence_similarities,
                "has_similarity": True,
            }, f, ensure_ascii=False)
        print(f"Saved similarity results to {out_path}")

    del model
    torch.cuda.empty_cache()


def build_argparser():
    parser = argparse.ArgumentParser(description="COCO RAG decode with token similarity injection")
    parser.add_argument("--model_path", type=str, default="Your Path", help="Model path")
    parser.add_argument("--data_path", type=str, default="Your Path", help="Data JSONL")
    parser.add_argument("--output_path", type=str, default="Your Path", help="Output prefix")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--decode_depth", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Fusion/DSAB args (active in this dual-stream script)
    parser.add_argument("--fusion_mode", type=str, default="logits_add", choices=["logits_add", "poe", "dsab"], help="Fusion mode between ctx (with context) and pri (no context)")
    parser.add_argument("--fusion_alpha", type=float, default=0.7, help="Weight for ctx in logits_add mode; pri weight is 1-alpha")
    parser.add_argument("--poe_beta", type=float, default=1.0, help="Weight for pri log-probs in poe mode")
    parser.add_argument("--dsab_stage", type=str, default="pre_warpers", choices=["strict", "pre_warpers"], help="DSAB application stage")
    parser.add_argument("--dsab_lite", action="store_true", help="Enable DSAB-lite dynamic alpha mixing")
    parser.add_argument("--dsab_alpha_gamma", type=float, default=1.0, help="DSAB alpha slope")
    parser.add_argument("--dsab_alpha_min", type=float, default=0.2, help="DSAB alpha min bound")
    parser.add_argument("--dsab_alpha_max", type=float, default=0.9, help="DSAB alpha max bound")
    parser.add_argument("--dsab_alpha_use_d2", action="store_true", help="Use D2 term in DSAB alpha mapping")
    parser.add_argument("--dsab_mix_space", type=str, default="logits", choices=["logits", "logprob"], help="Mixing space for DSAB-lite")
    parser.add_argument("--dsab_use_safeguards", action="store_true", help="[placeholder] DSAB safeguards (not fully wired)")
    # Sentence-similarity gain coupling (match base script naming)
    parser.add_argument("--ac_gain_gamma", type=float, default=1.0, help="Scale for sentence gain (1-AC) term added to DSAB alpha mapping")
    # Zero-diff & debug flags to mirror base script (internally we always step-emulate to enable similarity)
    parser.add_argument("--zerodiff_ctx_generate", action="store_true", help="[parity] Accept flag; internally step-emulate to inject similarity")
    parser.add_argument("--zerodiff_step", action="store_true", help="[parity] Emulate generate() step-by-step (default path)")
    parser.add_argument("--suppress_image_token", action="store_true", help="Suppress <image>-like tokens via custom logits processor")
    parser.add_argument("--debug_print_gen_config", action="store_true", help="Print effective generation_config key fields")
    parser.add_argument("--debug_compare_generate", action="store_true", help="[parity] Accept flag; not executed here")
    parser.add_argument("--debug_topk_k", type=int, default=5, help="Top-k to print when comparing per-step logits (parity)")
    return parser


def main():
    args = build_argparser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    evaluate(args)


if __name__ == "__main__":
    main()
