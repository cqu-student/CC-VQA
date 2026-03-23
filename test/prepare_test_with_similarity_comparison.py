#!/usr/bin/env python3
"""
带相似度的测试数据预处理脚本：基于 prepare_rag_sft_comparison_data.py
- 保留评测对比所需的 question/response 字段
- 使用 BLIP2 对 section 进行重排序
- 使用 vLLM 生成中间推理（<reason>）与总结（<think>）
- 对“最终总结的内容”（final_text_article）按句子计算相似度，并输出 sentence_similarities
    （此处已改为使用 EVA-CLIP 计算“改写后的问题(refined question)”与“每个句子(sentence)”的文本-文本相似度）
    （总结阶段：让模型将原始问题改写为“实体更清晰”的问题，输出在 <question></question> 中，并用于最终 query）
"""

import json
import os
import sys
import yaml
import math
import random
import argparse
import torch
from PIL import Image
from tqdm import tqdm
import logging
import re
import io
import base64
from openai import OpenAI

# 添加 EchoSight 路径到 Python 路径
echosight_path = "/EchoSight"
if echosight_path not in sys.path:
    sys.path.insert(0, echosight_path)

from retriever import ClipRetriever
from answer_generator import reconstruct_wiki_sections
from lavis.models import load_model_and_preprocess

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_blip2_reranker(qformer_ckpt_path, device="cuda"):
    """加载BLIP2 reranker模型"""
    logger.info("Loading BLIP2 reranker model...")
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_reranker", model_type="pretrain", is_eval=True, device=device
    )
    checkpoint = torch.load(qformer_ckpt_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint, strict=False)
    blip_model = blip_model.half()
    blip_model.use_vanilla_qformer = True
    logger.info(f"Missing keys: {msg.missing_keys}")
    from data_utils import targetpad_transform
    preprocess = targetpad_transform(1.25, 224)
    return blip_model, vis_processors, txt_processors, preprocess

def compute_section_similarities_with_blip(
    image, question, sections, blip_model, txt_processors, preprocess, device="cuda"
):
    """使用BLIP2计算图像+问题与sections的相似度"""
    if not sections:
        return torch.tensor([])
    reference_image = preprocess(image).to(device).unsqueeze(0)
    qformer_articles = [txt_processors["eval"](section) for section in sections]
    with torch.cuda.amp.autocast():
        fusion_embs = blip_model.extract_features(
            {"image": reference_image, "text_input": question},
            mode="multimodal",
        )["multimodal_embeds"]
        all_scores = []
        batch_size = 300
        for i in range(0, len(qformer_articles), batch_size):
            batch = qformer_articles[i:i + batch_size]
            article_embs = blip_model.extract_features(
                {"text_input": batch},
                mode="text",
            )["text_embeds_proj"][:, 0, :]
            scores = torch.matmul(
                article_embs.unsqueeze(1).unsqueeze(1),
                fusion_embs.permute(0, 2, 1),
            ).squeeze()
            if len(scores.shape) > 1:
                scores, _ = scores.max(-1)
            all_scores.append(scores)
        return torch.cat(all_scores, dim=0) if all_scores else torch.tensor([])

def extract_reason_from_response(response_text):
    """提取 <reason></reason> 内容"""
    if not response_text:
        return ""
    matches = re.findall(r'<reason>(.*?)</reason>', response_text, re.DOTALL)
    return matches[0].strip() if matches else response_text.strip()

def extract_think_from_response(response_text):
    """提取 <think></think> 内容"""
    if not response_text:
        return ""
    matches = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
    return matches[0].strip() if matches else response_text.strip()

def extract_question_from_response(response_text):
    """提取 <question></question> 内容（用于实体更清晰的问题改写）"""
    if not response_text:
        return ""
    matches = re.findall(r'<question>(.*?)</question>', response_text, re.DOTALL)
    return matches[0].strip() if matches else response_text.strip()

# -------- vLLM 工具 --------
def pil_to_base64_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def vllm_chat_completion(client: OpenAI, model: str, prompt: str, image: Image.Image | None = None,
                         temperature: float = 0.7, max_new_tokens: int = 512) -> str:
    if image is not None:
        img_b64 = pil_to_base64_jpeg(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }]
    else:
        messages = [{"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error: {e}"

def call_qwen2vl_with_prompt(model, processor, prompt, image=None, device="cuda", max_new_tokens=512):
    """兼容旧接口名，代理到 vLLM OpenAI 接口"""
    client: OpenAI = model
    served_model: str = processor
    return vllm_chat_completion(
        client=client,
        model=served_model,
        prompt=prompt,
        image=image,
        temperature=0.7,
        max_new_tokens=max_new_tokens,
    )

def split_text_into_sentences(text: str):
    """将文本切分为句子，过滤过短句子"""
    if not text or not text.strip():
        return []
    # 先按换行拆分，再按句号/问号/感叹号进一步拆
    parts = []
    for block in text.strip().splitlines():
        block = block.strip()
        if not block:
            continue
        parts.extend(re.split(r'[.!?]+\s+', block))
    sentences = [s.strip() for s in parts if s.strip() and len(s.strip()) > 10]
    return sentences

def find_answer_containing_sections(sections, answer, threshold=0.3):
    """
    搜索包含答案信息的sections
    
    Args:
        sections: section列表
        answer: 目标答案
        threshold: 匹配阈值
        
    Returns:
        list: 包含答案的section索引
    """
    answer_sections = []
    if not answer or not isinstance(answer, str):
        return []
    answer_lower = answer.lower().strip()
    
    # 如果答案是数字，也尝试匹配数字格式
    answer_variants = [answer_lower]
    if answer_lower.replace('.', '').replace(',', '').isdigit():
        # 数字的不同格式
        num_str = answer_lower.replace(',', '')
        answer_variants.extend([num_str, answer_lower.replace(',', '')])
    
    for i, section in enumerate(sections):
        section_lower = section.lower()
        
        # 1. 直接包含检查
        for variant in answer_variants:
            if variant in section_lower:
                answer_sections.append(i)
                break
        else:
            # 2. 词汇重叠检查（适用于复杂答案）
            answer_words = set(answer_lower.split())
            section_words = set(section_lower.split())
            
            if len(answer_words) > 0:
                overlap_ratio = len(answer_words.intersection(section_words)) / len(answer_words)
                if overlap_ratio >= threshold:
                    if i not in answer_sections:
                        answer_sections.append(i)
    
    return answer_sections

def ensure_answer_sections_included_multiple(
    unique_metas, final_scores, answer_list, top_sections=5
):
    """
    确保包含答案的sections被包含在最终选择中（支持多个候选答案）
    
    Args:
        unique_metas: 所有去重后的section元信息列表
        final_scores: 对应的融合后分数 (torch.Tensor)
        answer_list: 候选答案列表（用于搜索相关sections）
        top_sections: 最终选择的section数量
        
    Returns:
        tuple: (selected_metas, selected_sections_texts, final_scores_for_selected)
    """
    all_sections = [m["section_text"] for m in unique_metas]
    
    # 如果answer_list无效，则按分数直接取top
    if not answer_list or not isinstance(answer_list, list):
        logger.warning(f"Invalid answer_list: {answer_list}, using original top-k logic")
        sorted_scores, sorted_indices = torch.sort(final_scores, descending=True)
        top_indices = sorted_indices[:top_sections].cpu().numpy()
        
        selected_metas = [unique_metas[i] for i in top_indices]
        for i, idx in enumerate(top_indices):
            selected_metas[i]["final_score"] = float(final_scores[idx].item())
        selected_texts = [m["section_text"] for m in selected_metas]
        selected_scores = sorted_scores[:top_sections].cpu().numpy().tolist()
        return selected_metas, selected_texts, selected_scores

    # 找到包含任意候选答案的sections
    all_answer_section_indices = set()
    valid_answers = [str(a).strip() for a in answer_list if a and isinstance(a, str)]

    for answer in valid_answers:
        answer_indices = find_answer_containing_sections(all_sections, answer)
        all_answer_section_indices.update(answer_indices)
    
    answer_section_indices = list(all_answer_section_indices)
    
    # 按原始分数排序
    sorted_scores, sorted_indices = torch.sort(final_scores, descending=True)

    # 分离包含答案的sections和其他sections
    answer_metas, other_metas = [], []
    
    # 使用原始索引来构建列表
    for i in range(len(unique_metas)):
        meta = unique_metas[i]
        meta["final_score"] = float(final_scores[i].item())
        if i in answer_section_indices:
            answer_metas.append(meta)
        else:
            other_metas.append(meta)
            
    # 按分数排序
    answer_metas.sort(key=lambda m: m["final_score"], reverse=True)
    other_metas.sort(key=lambda m: m["final_score"], reverse=True)

    # 确定要选择的答案sections数量（最多选择一半，至少选择1个）
    num_answer_sections = min(len(answer_metas), max(1, top_sections // 2))
    selected_answer_metas = answer_metas[:num_answer_sections]
    
    # 剩余位置数量
    remaining_slots = top_sections - num_answer_sections
    selected_other_metas = other_metas[:remaining_slots]
    
    # 合并所有选中的sections
    final_selected_metas = selected_answer_metas + selected_other_metas
    
    # 随机打乱顺序
    random.shuffle(final_selected_metas)
    
    final_texts = [m["section_text"] for m in final_selected_metas]
    final_scores_for_selected = [m["final_score"] for m in final_selected_metas]
    
    logger.info(f"Selected {num_answer_sections} answer-containing sections out of {len(final_selected_metas)} total sections")
    
    return final_selected_metas, final_texts, final_scores_for_selected

def process_single_example(
    example, 
    image_root, 
    retriever, 
    blip_model, 
    txt_processors, 
    preprocess,
    answer_generator_model=None,  # OpenAI 客户端
    answer_processor=None,        # vLLM served model 名
    top_k=20,
    top_sections=3,
    alpha_1=0.5,
    device="cuda"
):
    """处理单个样本，输出包含最终文本按句子相似度的测试样本"""
    try:
        # 1. 加载图像
        image = None
        image_path_to_save = None
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            if not os.path.exists(image_path):
                if image_path.endswith('.JPEG'):
                    image_path = os.path.join(image_root, example['image'].replace('.JPEG', '.jpg'))
                elif image_path.endswith('.jpg'):
                    image_path = os.path.join(image_root, example['image'].replace('.jpg', '.JPEG'))
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                image_path_to_save = os.path.abspath(image_path)
                w, h = image.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if image is None:
            return None

        # 2. 取问题与答案
        question = example['question']
        answer_field_candidates = ['answer_eval', 'answer']
        answer = None
        for field in answer_field_candidates:
            if field in example and example[field]:
                answer = example[field]
                break
        if answer is None:
            logger.warning(f"No valid answer field found in example: {list(example.keys())}")
            return None

        # 2.1 传递数据集中自带的 ground_truth（若存在），规范化为 list[str]
        ground_truth_list = None
        if 'ground_truth' in example:
            gt_val = example.get('ground_truth')
            if isinstance(gt_val, list):
                ground_truth_list = [str(x).strip() for x in gt_val if x is not None and str(x).strip()]
            elif gt_val is not None:
                s = str(gt_val).strip()
                if s:
                    ground_truth_list = [s]

        # 3. 检索 section（保留元信息，包含 image_url 与 section 字段）
        retrieved_sections_meta = []
        top_retrieval = retriever.retrieve_image_faiss(image, top_k=top_k)
        if top_retrieval:
            for item in top_retrieval:
                entry = item["kb_entry"]
                title = entry.title
                # section_index -> image_urls 列表
                secidx_to_image_urls = {}
                for img_idx, sec_idx in enumerate(entry.image_section_indices):
                    if sec_idx is None:
                        continue
                    try:
                        sec_idx_int = int(sec_idx)
                    except Exception:
                        continue
                    secidx_to_image_urls.setdefault(sec_idx_int, []).append(entry.image_urls[img_idx])

                for sec_idx, sec_title in enumerate(entry.section_titles):
                    # 过滤参考/外链
                    if (
                        isinstance(sec_title, str)
                        and (
                            "external links" in sec_title.lower()
                            or "references" in sec_title.lower()
                            or "external link" in sec_title.lower()
                            or "reference" in sec_title.lower()
                        )
                    ):
                        continue
                    # 构造文本
                    if sec_idx < len(entry.section_texts):
                        sec_text_raw = entry.section_texts[sec_idx]
                    else:
                        sec_text_raw = ""
                    section_text = (
                        f"{sec_text_raw}"
                    ).strip()
                    retrieved_sections_meta.append({
                        "section_text": section_text,
                        "section_title": sec_title,
                        "section_index": sec_idx,
                        "page_title": title,
                        "wiki_url": entry.url,
                        "kb_index": item.get("knowledge_base_index"),
                        "retrieval_similarity": float(item.get("similarity", 0.0)),
                        "retrieval_image_url": item.get("image_url"),
                        "image_urls_for_section": secidx_to_image_urls.get(sec_idx, []),
                        "source": "retrieved",
                    })

        # 去重：按 section_text
        seen_texts = set()
        unique_sections_meta = []
        for meta in retrieved_sections_meta:
            text_key = meta["section_text"].strip()
            if text_key and text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_sections_meta.append(meta)

        # 3.5 优先从 entity_context 中选取“包含真实答案”的 section（若存在）
        used_gt_section = False
        selected_metas = []
        selected_sections_texts = []

        def _normalize_text(t: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (t or "").lower())).strip()

        def split_ground_truth_sections(entity_context: str):
            if not entity_context or not str(entity_context).strip():
                return []
            # 常见格式：用 '##' 分段；若没有，则退化为整篇
            parts = str(entity_context).split('##')
            out = []
            for p in parts:
                s = (p or "").strip()
                if s:
                    out.append(s)
            if not out:
                out = [str(entity_context).strip()]
            return out

        # entity_context = example.get('entity_context')
        # if entity_context is not None:
        #     gt_sections = split_ground_truth_sections(entity_context)
        #     # 构造匹配候选：answer 优先；如存在 ground_truth_list 也一并纳入
        #     match_texts = []
        #     if answer is not None and str(answer).strip():
        #         match_texts.append(str(answer))
        #     if isinstance(ground_truth_list, list):
        #         match_texts.extend(ground_truth_list)
        #     match_texts = [m for m in match_texts if m and str(m).strip()]

        #     if gt_sections and match_texts:
        #         norm_matches = [_normalize_text(str(m)) for m in match_texts]
        #         chosen_section = None
        #         for sec in gt_sections:
        #             ns = _normalize_text(sec)
        #             if any(nm and (nm in ns) for nm in norm_matches):
        #                 chosen_section = sec
        #                 break
        #         if chosen_section:
        #             # 构造与检索保持一致的 meta 结构
        #             gt_meta = {
        #                 "section_text": chosen_section,
        #                 "section_title": None,
        #                 "section_index": None,
        #                 "page_title": None,
        #                 "wiki_url": None,
        #                 "kb_index": None,
        #                 "retrieval_similarity": 1.0,
        #                 "retrieval_image_url": None,
        #                 "image_urls_for_section": [],
        #                 "source": "ground_truth",
        #                 "contains_answer": True,
        #             }
        #             selected_metas = [gt_meta]
        #             selected_sections_texts = [chosen_section]
        #             used_gt_section = True

        # 4. BLIP2 重排序（如未命中 GT section 时才进行）
        # 为后续强制包含答案段落，记录每个候选的分数（与排序使用一致）
        scores_for_unique = []
        has_scores = False
        if (not used_gt_section) and unique_sections_meta:
            blip_scores = compute_section_similarities_with_blip(
                image, question, [m["section_text"] for m in unique_sections_meta], blip_model, txt_processors, preprocess, device
            )
            if len(blip_scores) > 0 and len(blip_scores) == len(unique_sections_meta):
                alpha_2 = 1 - alpha_1
                retrieval_tensor = torch.tensor([m["retrieval_similarity"] for m in unique_sections_meta], device=device, dtype=blip_scores.dtype)
                if blip_scores.device != retrieval_tensor.device:
                    retrieval_tensor = retrieval_tensor.to(blip_scores.device)
                final_scores = alpha_1 * retrieval_tensor + alpha_2 * blip_scores

                # 不强制包含答案：按融合分数直接选择 Top-K sections
                sorted_scores, sorted_indices = torch.sort(final_scores, descending=True)
                k = min(top_sections, len(unique_sections_meta))
                top_indices = sorted_indices[:k].cpu().numpy().tolist()

                selected_metas = [unique_sections_meta[i] for i in top_indices]
                for i, idx in enumerate(top_indices):
                    selected_metas[i]["final_score"] = float(final_scores[idx].item())
                selected_sections_texts = [m["section_text"] for m in selected_metas]
                scores_for_selected = sorted_scores[:k].detach().cpu().numpy().tolist()
                has_scores = True

            elif len(blip_scores) > 0:
                # Fallback to only blip_scores if retrieval similarity is missing
                final_scores = blip_scores

                # 不强制包含答案：按 BLIP 分数直接选择 Top-K sections
                sorted_scores, sorted_indices = torch.sort(final_scores, descending=True)
                k = min(top_sections, len(unique_sections_meta))
                top_indices = sorted_indices[:k].cpu().numpy().tolist()

                selected_metas = [unique_sections_meta[i] for i in top_indices]
                for i, idx in enumerate(top_indices):
                    selected_metas[i]["final_score"] = float(final_scores[idx].item())
                selected_sections_texts = [m["section_text"] for m in selected_metas]
                scores_for_selected = sorted_scores[:k].detach().cpu().numpy().tolist()
                has_scores = True
            else:
                # No scores at all, select top-k based on retrieval order
                selected_metas = unique_sections_meta[:top_sections]
                selected_sections_texts = [m["section_text"] for m in selected_metas]
        
        # 4.5 (已合并到步骤4)
        
        # 5. 三步 prompt（vLLM）
        all_reasons = []
        if answer_generator_model is not None and answer_processor is not None:
            prompt1 = f"Here is the question:{question}, Give your answer and put the feature or reason for the answer related to the image in <reason></reason>."
            response1 = call_qwen2vl_with_prompt(answer_generator_model, answer_processor, prompt1, image, device)
            reason1 = extract_reason_from_response(response1)
            all_reasons.append(reason1)
            logger.info(f"Step 1 reason extracted: {reason1[:100]}...")
        else:
            all_reasons.append("Initial model reasoning based on image features")

        section_reasons = []
        for i, section in enumerate(selected_sections_texts):
            if answer_generator_model is not None and answer_processor is not None:
                prompt2 = f"Here is the question: {question}, Here is the selected section: {section}, Give your answer and put the feature or reason for the answer related to the image in <reason></reason>."
                response2 = call_qwen2vl_with_prompt(answer_generator_model, answer_processor, prompt2, image, device)
                reason2 = extract_reason_from_response(response2)
                section_reasons.append(reason2)
                logger.info(f"Step 2 section {i+1} reason extracted: {reason2[:100]}...")
            else:
                section_reasons.append(f"Reasoning based on section {i+1}: {section[:50]}...")
        all_reasons.extend(section_reasons)

        if len(all_reasons) >= 4 and answer_generator_model is not None and answer_processor is not None:
            reasons_text = "\n".join([f"Reason {i+1}: {reason}" for i, reason in enumerate(all_reasons)])
            prompt3 = (
                f"Here is the question: {question} Below are the reasons supporting the answer derived from the retrieved information:\n"
                f"{reasons_text}\n"
                f"Identify the key features that distinguish the aforementioned categories and extract the features that need attention in <think></think>, without detailed illustration or answer."
            )
            response3 = call_qwen2vl_with_prompt(answer_generator_model, answer_processor, prompt3, None, device)
            final_reason = extract_think_from_response(response3)
            logger.info(f"Step 3 final reasoning extracted: {final_reason[:100]}...")
            # final_text_article = f" Here is the retrieved information:\n" + "\n".join(selected_sections_texts)
            final_text_article = f"Here is the feature to focus on: \n<think>{final_reason}</think>\n Here is the retrieved information:\n" + "\n\n".join(selected_sections_texts)
        else:
            final_text_article = "\n\n".join(selected_sections_texts)

        # 3.5 改写问题为“实体更清晰”的版本，并替换用于最终 query（保留原始问题以便追踪）
        refined_question = question
        if answer_generator_model is not None and answer_processor is not None:
            prompt_q = (
                "Please refer to the given image and rewrite the question so that entities and attributes are explicit, pronouns are disambiguated, and the wording is more specific.\n"
                "Keep the original intent and scope unchanged; do not add new information.\n"
                "Only output the rewritten question wrapped in <question></question>.\n\n"
                f"Original question:\n{question}"
            )
            response_q = call_qwen2vl_with_prompt(
                answer_generator_model, answer_processor, prompt_q, image, device, max_new_tokens=128
            )
            rq = extract_question_from_response(response_q)
            if rq:
                refined_question = rq

        # 6. 对"最终总结的内容"按句子计算相似度（使用 EVA-CLIP：refined_question vs sentence 文本-文本相似度）
        sentence_similarities = []
        sentences = split_text_into_sentences(final_text_article)
        if sentences:
            try:
                sims = retriever.similarity_section_text(refined_question, sentences)
                # sims 为张量，按顺序对应每个句子
                sentence_similarities = [float(s.item()) for s in sims]
            except Exception as e:
                logger.warning(f"Failed to compute sentence similarities with EVA-CLIP: {e}")
                sentence_similarities = [0.5] * len(sentences)
        else:
            sentence_similarities = [0.5]

        # 7. 构建样本（评测对比用），并输出相似度信息
        user_content = f"<image>Context: \nHere is the Question: {refined_question}\n{final_text_article}, \nShort answer:"
        sample_data = {
            "query": user_content,
            "question": refined_question,
            "response": answer,
            # 附带最终文本与句子相似度（用于后续分析/训练）
            "retrieved_content": final_text_article,
            "sentence_similarities": sentence_similarities,
            # 保存数据划分，来源于原始样本或 YAML 推断
            "data_split": example.get("data_split", "unknown"),
        }
        # 若原始数据自带 ground_truth，则一并保留（按 list[str] 规范化）
        if ground_truth_list is not None:
            sample_data["ground_truth"] = ground_truth_list
        # 保留原始问题以便追踪
        if refined_question != question:
            sample_data["original_question"] = question
        if image_path_to_save:
            sample_data["images"] = [image_path_to_save]
        sample_data["extracted_reasons"] = all_reasons
        # 附带选中 section 的元信息（包含 image_url 与 section 字段）
        if selected_metas:
            sample_data["selected_sections"] = [
                {
                    "section_text": m.get("section_text"),
                    "section_title": m.get("section_title"),
                    "section_index": m.get("section_index"),
                    "page_title": m.get("page_title"),
                    "wiki_url": m.get("wiki_url"),
                    "retrieval_similarity": m.get("retrieval_similarity"),
                    "blip_similarity": m.get("blip_similarity"),
                    "final_score": m.get("final_score"),
                    "image_urls_for_section": m.get("image_urls_for_section", []),
                    "retrieval_image_url": m.get("retrieval_image_url"),
                    "source": m.get("source", "retrieved"),
                }
                for m in selected_metas
            ]
        return sample_data

    except Exception as e:
        logger.warning(f"Error processing example: {e}. Skipping sample.")
        return None

def load_data_from_yaml(data_path, max_samples=None):
    """从YAML配置文件加载数据"""
    list_data_dict = []
    if data_path.endswith(".yaml"):
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")
            for data in datasets:
                data_dict = data.get("data")
                json_path = data_dict.get("json_path")
                # 数据集级别 split：优先使用 YAML 标注，其次根据文件名推断
                ds_level_split = (
                    data.get("split")
                    or data_dict.get("split")
                    or ("train" if "train" in str(json_path).lower() else None)
                )
                if not ds_level_split:
                    name_l = str(json_path).lower()
                    if any(k in name_l for k in ["val", "valid", "validation", "dev"]):
                        ds_level_split = "val"
                    elif "test" in name_l or "testing" in name_l:
                        ds_level_split = "test"
                sampling_strategy = data_dict.get("sampling_strategy", "all")
                sampling_number = data_dict.get("sampling_number", None)
                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            # 样本级别 split：样本内 > YAML 数据集级 > 文件名推断 > unknown
                            sample_split = (
                                obj.get("data_split")
                                or obj.get("split")
                                or ds_level_split
                            )
                            if not sample_split:
                                sample_split = "unknown"
                            obj["data_split"] = sample_split
                            cur_data_dict.append(obj)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")
                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]
                logger.info(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                list_data_dict.extend(cur_data_dict)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")
    if max_samples and len(list_data_dict) > max_samples:
        list_data_dict = list_data_dict[:max_samples]
        logger.info(f"Limited to {max_samples} samples for faster processing")
    return list_data_dict

def main():
    parser = argparse.ArgumentParser(description='Prepare RAG test data with sentence-level similarity on final summary')
    parser.add_argument('--data_path', type=str, required=True, help='Input data YAML config path')
    parser.add_argument('--output_path', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--image_root', type=str, required=True, help='Image root directory')
    parser.add_argument('--kb_path', type=str, required=True, help='Knowledge base JSON path')
    parser.add_argument('--faiss_index_path', type=str, required=True, help='FAISS index path')
    parser.add_argument('--qformer_ckpt_path', type=str, required=True, help='QFormer checkpoint path')
    parser.add_argument('--answer_model_path', type=str, default=None, help='Deprecated: local Qwen2.5-VL path (unused)')
    parser.add_argument('--top_k', type=int, default=3, help='Top-k retrieval (default: 3)')
    parser.add_argument('--top_sections', type=int, default=5, help='Top sections to select (default: 5)')
    parser.add_argument('--alpha_1', type=float, default=0.5, help='Weight for retrieval similarity (default: 0.5)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--retriever_model', type=str, default='eva-clip', help='Retriever model type (default: eva-clip)')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers (default: 1)')
    parser.add_argument('--worker_id', type=int, default=0, help='Worker ID for distributed processing (default: 0)')
    # vLLM 相关参数
    parser.add_argument('--vllm_base_url', type=str, default='', help='vLLM OpenAI base URL')
    parser.add_argument('--vllm_model', type=str, required=True, help='vLLM served model name (id from /v1/models)')
    parser.add_argument('--vllm_api_key', type=str, default='EMPTY', help='API key for vLLM server')
    args = parser.parse_args()

    # 设置GPU设备
    if torch.cuda.is_available():
        if args.num_workers > 1:
            gpu_id = args.worker_id % torch.cuda.device_count()
            torch.cuda.set_device(gpu_id)
            args.device = f"cuda:{gpu_id}"
            logger.info(f"Worker {args.worker_id} using GPU {gpu_id}")
        else:
            gpu_id = torch.cuda.current_device()
            logger.info(f"Using GPU {gpu_id}")

    # 切换到 EchoSight 目录以确保相对导入正常工作
    echosight_dir = "/EchoSight"
    original_dir = os.getcwd()
    os.chdir(echosight_dir)
    logger.info(f"Changed working directory to: {echosight_dir}")

    try:
        logger.info("Starting RAG test data preparation (with sentence-level similarity on final summary)...")
        logger.info(f"Config: top_k={args.top_k}, top_sections={args.top_sections}, alpha_1={args.alpha_1}")

        # 1. 加载数据
        raw_data = load_data_from_yaml(args.data_path, args.max_samples)
        logger.info(f"Loaded {len(raw_data)} samples")

        # 多 worker 分片
        if args.num_workers > 1:
            total = len(raw_data)
            per = (total + args.num_workers - 1) // args.num_workers
            start = args.worker_id * per
            end = min((args.worker_id + 1) * per, total)
            raw_data = raw_data[start:end]
            logger.info(f"Worker {args.worker_id} processing samples {start}-{end-1} ({len(raw_data)} samples)")
            base_name, ext = os.path.splitext(args.output_path)
            args.output_path = f"{base_name}_worker_{args.worker_id}{ext}"

        # 2. 初始化检索器
        if torch.cuda.is_available():
            if args.num_workers > 1:
                gpu_choice = args.worker_id % torch.cuda.device_count()
                logger.info(f"Worker {args.worker_id} using GPU {gpu_choice} for FAISS index")
            else:
                gpu_choice = torch.cuda.current_device()
                logger.info(f"Using GPU {gpu_choice} for retriever")
        else:
            gpu_choice = 0
            logger.warning("CUDA not available, using GPU 0 as default")

        retriever = ClipRetriever(device=args.device, model=args.retriever_model)
        retriever.load_knowledge_base(args.kb_path)
        retriever.load_faiss_index(args.faiss_index_path, gpu_choice=gpu_choice)

        # 3. 加载 BLIP2
        blip_model, vis_processors, txt_processors, preprocess = load_blip2_reranker(
            args.qformer_ckpt_path, args.device
        )

        # 4. 初始化 vLLM 客户端
        vllm_client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        served_model = args.vllm_model
        logger.info(f"vLLM client ready. Using model: {served_model}")

        # 5. 处理数据
        processed = []
        for i, example in enumerate(tqdm(raw_data, desc=f"Worker {args.worker_id} processing examples")):
            item = process_single_example(
                example=example,
                image_root=args.image_root,
                retriever=retriever,
                blip_model=blip_model,
                txt_processors=txt_processors,
                preprocess=preprocess,
                answer_generator_model=vllm_client,
                answer_processor=served_model,
                top_k=args.top_k,
                top_sections=args.top_sections,
                alpha_1=args.alpha_1,
                device=args.device
            )
            if item:
                processed.append(item)
            if (i + 1) % 100 == 0:
                logger.info(f"Worker {args.worker_id} processed {i + 1}/{len(raw_data)} samples")

        # 6. 保存
        os.chdir(original_dir)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for it in processed:
                f.write(json.dumps(it, ensure_ascii=False) + '\n')
        logger.info(f"Worker {args.worker_id} completed! Processed {len(processed)} samples")
        logger.info(f"Output saved to: {args.output_path}")

        if args.num_workers > 1:
            logger.info(f"To merge all worker outputs, run:")
            base_name, ext = os.path.splitext(args.output_path.replace(f"_worker_{args.worker_id}", ""))
            logger.info(f"cat {base_name}_worker_*{ext} > {base_name}{ext}")

    finally:
        os.chdir(original_dir)
        logger.info(f"Restored working directory to: {original_dir}")

if __name__ == "__main__":
    main()
