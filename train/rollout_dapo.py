import os
import sys
from typing import Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import math
import random
import re
from collections import Counter
from transformers import AutoTokenizer, AutoConfig
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
import torch.distributed as dist
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out

from sample_lmdeploy import sample
from orm import orms


from utils.data_chunk import get_data_chunk
from utils.math_utils import equation, extract_last_boxed
from train.utils import get_config

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out
from accelerate import Accelerator, utils
import accelerate
from transformers import AutoTokenizer
import re

EXTRACT_FAILED="Can not extract the answer!"
def extract_final_boxed_answer(s: str):
    tag = r'boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def z_score_normalize(lst):
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    return [(x - mean) / std if std != 0 else 0 for x in lst]


def rollout_sampling(dataset_name, epoch, config, model_path, tokenizer, accelerator: Accelerator, reward_function=None, normalize=True, filter=True, mode="train", split_rank=False, reward_funcs="accuracy", data_cursor=0):
    # 顺序遍历模式：通过 data_cursor 参数记录数据集遍历位置
    # 每个epoch按顺序遍历 num_task_per_step * sampling_multiplier 的数据
    # 从 data_cursor 位置开始，返回时更新 cursor
    
    project_name = config.experiment.project
    reward = config.dataset.data_type

    # Parse reward_funcs (comma-separated string)
    reward_func_list = [func.strip() for func in reward_funcs.split(',') if func.strip()]
    
    # Parse reward_weights (comma-separated string or list)
    if not hasattr(config.training, 'reward_weights') or config.training.reward_weights is None:
        reward_weights = [1.0] * len(reward_func_list)
    elif isinstance(config.training.reward_weights, str):
        reward_weights = [float(w.strip()) for w in config.training.reward_weights.split(',') if w.strip()]
    else:
        reward_weights = list(config.training.reward_weights)
    
    if len(reward_weights) != len(reward_func_list):
        if accelerator.is_main_process:
            print(f"Warning: reward_weights length ({len(reward_weights)}) != reward_funcs length ({len(reward_func_list)})")
            print(f"Using default weights: all 1.0")
        reward_weights = [1.0] * len(reward_func_list)
    
    if accelerator.is_main_process:
        print(f"Using reward functions: {reward_func_list}")
        print(f"Using reward weights: {reward_weights}")

    # Initialize ORMs (keep all for backward compatibility)
    math_judge = orms['accuracy']()
    cosine_reward = orms['cosine'](cosine_max_len=config.rollout.max_token, accuracy_orm=math_judge)
    repetition_reward = orms['repetition']()
    format_reward = orms['format']()
    boxed_reward = orms['boxed']()
    binary_reward = orms['binary']()
    soft_overlong_reward = orms['soft_overlong'](config.rollout.max_token, config.rollout.max_token//2)
    
    # Initialize ORM instances for reward_func_list
    orm_instances = {}
    for func_name in reward_func_list:
        if func_name == 'accuracy':
            orm_instances[func_name] = math_judge
        elif func_name == 'cosine':
            orm_instances[func_name] = cosine_reward
        elif func_name == 'repetition':
            orm_instances[func_name] = repetition_reward
        elif func_name == 'format':
            orm_instances[func_name] = format_reward
        elif func_name == 'boxed':
            orm_instances[func_name] = boxed_reward
        elif func_name == 'binary':
            orm_instances[func_name] = binary_reward
        elif func_name == "soft_overlong":
            orm_instances[func_name] = soft_overlong_reward
        else:
            # Try to instantiate directly from orms
            try:
                orm_instances[func_name] = orms[func_name]()
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Warning: failed to initialize ORM '{func_name}': {e}")
                    
    def compute_reward_from_funcs(text, gt, token_ids=None):
        """Compute reward by calling each ORM function and summing results"""
        total_reward = 0.0
        reward_components = {}
        
        for func_name in reward_func_list:
            if func_name not in orm_instances:
                continue
                
            orm = orm_instances[func_name]
            try:

                score = orm(completions=[text], solution=[gt], response_token_ids=[token_ids], max_tokens=config.rollout.max_token)[0]
            
                reward_components[func_name] = score
                total_reward += score
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Warning: error computing {func_name} reward: {e}")
                reward_components[func_name] = 0.0
        
        return total_reward, reward_components

    backend_config = PytorchEngineConfig(
        dtype="float16",
        max_prefill_token_num=config.rollout.max_token * 64,
        cache_max_entry_count=0.9,
        dllm_block_length=config.rollout.block_size,
        dllm_denoising_steps=config.rollout.denoising_steps_per_block,
        dllm_unmasking_strategy=config.rollout.remasking_strategy,
        dllm_confidence_threshold=config.rollout.dynamic_threshold,
    )
    # config.
    if mode == "train":
        gen_config = GenerationConfig(
            top_p=config.rollout.top_p,
            top_k=config.rollout.top_k,
            temperature=config.rollout.temperature,
            do_sample=config.rollout.do_sample,
            max_new_tokens=config.rollout.max_token,
            min_new_tokens=128,
            skip_special_tokens=False,
            # random_seed=config.training.seed + epoch,
        )
    else:
        # 评测模式 gen_config 与 eval_lmdeploy.py 完全一致
        do_sample = config.evaluation.do_sample
        gen_config = GenerationConfig(
            n=1,
            top_p=config.evaluation.top_p,
            top_k=config.evaluation.top_k,
            temperature=config.evaluation.temperature,
            do_sample=do_sample,
            max_new_tokens=config.evaluation.max_token,
            min_new_tokens=128,
            skip_special_tokens=False,
            # random_seed=config.training.seed + epoch,
        )
    if accelerator.is_main_process:
        print(f"gen_config: {gen_config}")
        
    start_with_think = config.rollout.start_with_think

    # Load data (rank0 构建缓存，其余读取)
    if accelerator.is_main_process:
        dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")

    ds = dataset['train']
    prompts_all = ds['question']
    gts_all = ds['ground_truth_answer']

    reason_prompt = "<|im_start|>user\n{problem}\nPlease reason step by step, and put your final answer within $\\boxed{{}}$.<|im_end|>\n<|im_start|>assistant\n"

    if start_with_think:
        reason_prompt = reason_prompt + "<think>"

    # Build pool of all (idx, (prompt, gt))
    all_data = list(enumerate(zip(prompts_all, gts_all)))

    # How many tasks to finally keep (global target)
    if mode == "train":
        if config.rollout.num_task_per_step == -1:
            target_total = len(all_data)
        else:
            target_total = min(config.rollout.num_task_per_step, len(all_data))
    else:
        target_total = len(all_data)

    # 顺序遍历策略：每个epoch按顺序遍历数据集，不打乱
    # 从 data_cursor 位置开始，遍历完后从头循环
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    
    # 根据数据集确定采样倍数（考虑DAPO过滤率）
    if "GSM8K" in dataset_name:
        sampling_multiplier = 12
    elif "BigMath" in dataset_name:
        sampling_multiplier = 4
    else:
        sampling_multiplier = 3
    
    total_to_sample = target_total * sampling_multiplier
    
    if mode == "train":
        # 使用传入的 data_cursor
        start_cursor = data_cursor
        
        # 计算统计信息
        total_data_size = len(all_data)
        epochs_per_pass = math.ceil(total_data_size / total_to_sample) if total_to_sample > 0 else 1
        completed_passes = start_cursor // total_data_size
        current_position = start_cursor % total_data_size
        current_pass_progress = current_position / total_data_size * 100
        end_position = (start_cursor + total_to_sample) % total_data_size
        
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"顺序遍历数据集 - Epoch {epoch}")
            print(f"{'='*80}")
            print(f"  数据集: {dataset_name}")
            print(f"  数据集大小: {total_data_size}")
            print(f"  目标收集数: {target_total}")
            print(f"  采样倍数: {sampling_multiplier}x (考虑DAPO过滤)")
            print(f"  预计遍历: {total_to_sample} 条 ({target_total} * {sampling_multiplier})")
            print(f"  完整遍历需要: {epochs_per_pass} epochs")
            print(f"  已完成轮数: {completed_passes}")
            print(f"  全局cursor: {start_cursor} (数据集内位置: {current_position}, {current_pass_progress:.1f}%)")
            
            # 计算实际会访问的索引范围（考虑循环）
            if (start_cursor + total_to_sample) // total_data_size > start_cursor // total_data_size:
                # 会循环
                first_part_end = total_data_size
                second_part_end = end_position
                print(f"  本epoch访问: [{current_position}, {first_part_end}) + [0, {second_part_end}) (循环)")
            else:
                # 不会循环
                print(f"  本epoch访问: [{current_position}, {(current_position + total_to_sample) % total_data_size})")
            print(f"{'='*80}\n")
        
        # 按顺序索引，不打乱
        ordered_indices = list(range(len(all_data)))
        global_cursor = start_cursor
    else:
        # 评估模式不需要cursor
        ordered_indices = list(range(len(all_data)))
        global_cursor = 0
    
    def get_next_batch(batch_size):
        """按顺序循环获取下一批数据"""
        nonlocal global_cursor
        batch = []
        for _ in range(batch_size):
            # 循环索引
            idx = ordered_indices[global_cursor % len(ordered_indices)]
            batch.append(idx)
            global_cursor += 1
        return batch

    selected_data = []
    count_low = 0
    count_mid = 0
    count_high = 0
    # 统计 mean_correctness 恰为 0 和 1 的数量（覆盖整个采样过程）
    count_mean0 = 0
    count_mean1 = 0

    k_sample = getattr(config.rollout if mode == "train" else config.evaluation, 'num_response_per_task', 1)

    def process_batch(batch_indices):
        nonlocal count_low, count_mid, count_high, count_mean0, count_mean1
        if len(batch_indices) == 0:
            return []
        idx_pairs = [all_data[i] for i in batch_indices]
        _, prompt_gt_pairs = zip(*idx_pairs)
        batch_prompts, batch_gts = zip(*prompt_gt_pairs)
        prompts_list = [reason_prompt.format(problem=p) for p in batch_prompts]

        # 统一调用 sample 函数，都传 config 和 accelerator
        outputs = sample(model_path, backend_config, gen_config, prompts_list, k_sample, config, accelerator)
        
        expected = len(prompts_list) * k_sample
        actual = len(outputs)
        if actual == 0:
            return []
        if actual != expected:
            # 截断到整组
            usable = (actual // k_sample) * k_sample
            outputs = outputs[:usable]
            lost_tasks = len(prompts_list) - (usable // k_sample)
            if lost_tasks > 0:
                print(f"[Rank {rank}] Warning: process_batch lost {lost_tasks} tasks due to output mismatch (expected {expected}, got {actual}, usable {usable})")

        batch_selected = []
        for original_idx in range(0, len(outputs) // k_sample):
            question = batch_prompts[original_idx]
            prompt = prompts_list[original_idx]
            gt = batch_gts[original_idx]

            combined_texts = []
            combined_step_maps = []
            extracted_answers = []
            response_lengths = []
            correctness_list = []
            raw_rewards = []

            for i in range(k_sample):
                output_idx = original_idx * k_sample + i
                o = outputs[output_idx]
                text = o.text
                token_ids = o.token_ids
                step_map = o.step_map
                # assert "<|im_end|>" in text or "<|endoftext|>" in text or len(token_ids)>=config.rollout.max_token, f"len: {len(token_ids)}, eos: {token_ids[-1]}"
                
                combined_texts.append(text)
                combined_step_maps.append(step_map)
                extracted_answers.append(extract_final_boxed_answer(text))
                response_lengths.append(len(token_ids))

                if reward == "math":
                    correctness = math_judge(completions=[text], solution=[gt])[0]
                    correctness_list.append(correctness)

                    if mode == "train":
                        # 使用新的reward_funcs系统计算reward
                        total_reward, reward_components = compute_reward_from_funcs(
                            text=text, 
                            gt=gt, 
                            token_ids=token_ids
                        )
                        
                        # 记录每个reward function的分数（按reward_func_list的顺序）
                        reward_scores = [reward_components.get(func_name, 0.0) for func_name in reward_func_list]
                        raw_rewards.append(reward_scores)

                    else:
                        # 评测模式：使用 math_judge 的结果，与 eval_lmdeploy.py 保持一致
                        raw_rewards.append([correctness])
                else:
                    correctness_list.append(0)
                    raw_rewards.append([0.0] * len(reward_func_list))

            mean_correctness = sum(correctness_list) / len(correctness_list)
            if mean_correctness < 0.2:
                count_low += 1
            elif mean_correctness <= 0.8:
                count_mid += 1
            else:
                count_high += 1

            # 额外统计：恰为 0 或 1 的数量
            if mean_correctness == 0.0:
                count_mean0 += 1
            elif mean_correctness == 1.0:
                count_mean1 += 1

            # DAPO 关键筛选：仅保留 mean_correctness 不等于 0 或 1 的题目
            if mode == "train":
                if not (0.0 < mean_correctness < 1.0 and max(correctness_list) == 1.0):
                    continue
                # 筛掉答案一致比率占比大于0.8的题目
                # answer_counts = Counter(extracted_answers)
                # if answer_counts:
                #     most_common_answer, most_common_count = answer_counts.most_common(1)[0]
                #     most_common_ratio = most_common_count / len(extracted_answers)
                #     if most_common_ratio >= 0.8:
                #         continue

        
            # 如果过滤后没有剩余响应，跳过这个问题
            if len(combined_texts) == 0:
                continue
            
            actual_k = len(combined_texts)
            # raw_rewards 是二维列表: [[r1_f1, r1_f2, ...], [r2_f1, r2_f2, ...], ...]
            # 使用 reward_weights 加权求和
            raw_rewards_scalar = [sum(r * w for r, w in zip(x, reward_weights)) for x in raw_rewards]
            if mode == "train":
                if normalize:
                    rewards = z_score_normalize(raw_rewards_scalar)
                else:
                    rewards = raw_rewards_scalar
            else:
                rewards = raw_rewards_scalar

            # 计算 extract_reward: 提取成功为1，失败为0
            extract_rewards = [1.0 if ext != EXTRACT_FAILED else 0.0 for ext in extracted_answers]

            # 评测模式使用与 eval_lmdeploy.py 一致的数据格式，训练模式保持原有格式
            if mode == "eval":
                batch_selected.append({
                    "question": question,
                    "prompt": prompt,
                    "ground_truth_answer": gt,
                    "full_output": combined_texts,
                    "extracted_output": extracted_answers,
                    "response_length": response_lengths,
                    "rewards": rewards,
                    "extract_reward": extract_rewards,
                    "step_map": combined_step_maps,
                })
            else:
                batch_selected.append({
                    "question": [question] * actual_k,
                    "prompt": [prompt] * actual_k,
                    "response": combined_texts,
                    "ground_truth_answer": [gt] * actual_k,
                    "full_output": combined_texts,
                    "extracted_output": extracted_answers,
                    "response_length": response_lengths,
                    "correctness": correctness_list,
                    "rewards": rewards,
                    "raw_rewards": raw_rewards_scalar,
                    "reward_components": raw_rewards,
                    "extract_reward": extract_rewards,
                    "step_map": combined_step_maps,
                })
        print(f"[Rank {rank}] Processed {len(batch_indices)} tasks")
        return batch_selected


    if mode == "train":
        # 保持原来的循环采样逻辑，但改为顺序遍历
        def get_global_count():
            """获取全局已收集的数据数量"""
            accelerator.wait_for_everyone()
            local_count = torch.tensor(len(selected_data), device=accelerator.device)
            return accelerator.gather(local_count).sum().item()

        batch_size = config.rollout.num_task_per_step if config.rollout.num_task_per_step != -1 else len(all_data)
        round_num = 0
        
        # 循环采样直到达到目标数量
        while get_global_count() < target_total:
            round_num += 1
            # 每轮采样 sampling_multiplier 倍的任务量（考虑DAPO过滤）
            sample_size = batch_size * sampling_multiplier
            # 获取全局批次，然后按 rank 分片
            global_batch = get_next_batch(sample_size)
            local_batch = get_data_chunk(global_batch, world_size, rank)
            selected_data.extend(process_batch(local_batch))
            
            global_selected = get_global_count()
            if accelerator.is_main_process:
                print(f"Round {round_num}: global selected {global_selected} / target {target_total}")
        
        final_global = get_global_count()
        if accelerator.is_main_process:
            print(f"Before truncation: global total {final_global}, target {target_total}")
        accelerator.print(f"[Rank {rank}] DAPO: local {len(selected_data)}, global {final_global} / target {target_total}")
    
    if mode == "train":
        # 训练模式：gather counts 和 data，进行 DAPO 统计和截断
        all_counts = gather_object([count_low, count_mid, count_high, count_mean0, count_mean1])
        all_data_collected = gather_object(selected_data)

        # Main process: 展平并截断
        if accelerator.is_main_process:
            # 展平多 rank 数据
            if isinstance(all_data_collected[0], list):
                all_data_collected = [item for rank_data in all_data_collected for item in rank_data]
            
            # 截断到 target_total
            if len(all_data_collected) > target_total:
                original_count = len(all_data_collected)
                random.shuffle(all_data_collected)
                all_data_collected = all_data_collected[:target_total]
                print(f"Randomly sampled {target_total} from {original_count} collected tasks")
            
            # 统计
            total_counts = [sum(all_counts[i::5]) for i in range(5)]
            total_low, total_mid, total_high, total_mean0, total_mean1 = total_counts
            total = total_low + total_mid + total_high
            
            if total > 0:
                print(f"\n{'='*60}")
                print(f"Global Correctness Distribution Statistics")
                print(f"Epoch: {epoch} | Target: {target_total} | Final Collected: {len(all_data_collected)}")
                print(f"  <0.2:     {total_low:5d} ({total_low/total*100:5.1f}%)")
                print(f"  0.2-0.8:  {total_mid:5d} ({total_mid/total*100:5.1f}%)")
                print(f"  >0.8:     {total_high:5d} ({total_high/total*100:5.1f}%)")
                print(f"  mean==0:  {total_mean0:5d} ({total_mean0/total*100:5.1f}%)")
                print(f"  mean==1:  {total_mean1:5d} ({total_mean1/total*100:5.1f}%)")
                print(f"{'='*60}\n")
        else:
            all_data_collected = None
        
        # Broadcast 截断后的数据到所有 rank
        from accelerate.utils import broadcast_object_list
        all_data_collected = broadcast_object_list([all_data_collected], from_process=0)[0]
        # 每个rank获取部分切片
        local_data_collected = get_data_chunk(all_data_collected, world_size, rank)
        # Log collected data size for each rank
        accelerator.print(f"[Rank {rank}] Collected {len(local_data_collected)} tasks after truncation and distribution.")
        
        # 返回数据和更新后的cursor
        updated_cursor = global_cursor
        if accelerator.is_main_process:
            print(f"Cursor updated: {start_cursor} -> {updated_cursor} (advanced {updated_cursor - start_cursor})")
        
        if not split_rank:
            return local_data_collected, updated_cursor
        else:
            return all_data_collected, updated_cursor



def rollout_sampling_eval(dataset_name, epoch, config, model_path, tokenizer, accelerator, reward_function=None, normalize=True, filter=True, mode="train"):
    
    # epoch = 1
    # 每轮控制采样的子集数据集不一样，需要设置不同的随机种子
    if config.training.seed is not None:
        random.seed(config.training.seed)

    project_name = config.experiment.project
    reward = config.dataset.data_type
    math_judge = orms['accuracy']()
    cosine_reward = orms['cosine'](cosine_max_len=config.rollout.max_token, accuracy_orm=math_judge)
    repetition_reward = orms['repetition']()
    format_reward = orms['format']()

    backend_config = PytorchEngineConfig(
        dtype="float16",
        max_prefill_token_num=config.rollout.max_token*64,
        cache_max_entry_count=0.9,
        dllm_block_length=config.rollout.block_size,
        dllm_denoising_steps=config.rollout.denoising_steps_per_block,
        dllm_unmasking_strategy=config.rollout.remasking_strategy,
        dllm_confidence_threshold=config.rollout.dynamic_threshold,
    )

    gen_config = GenerationConfig(
        top_p=config.evaluation.top_p,
        top_k=config.evaluation.top_k,
        temperature=config.evaluation.temperature,
        do_sample=config.evaluation.do_sample,
        min_new_tokens=128, # 防止突然早停
        max_new_tokens=config.evaluation.max_token,
        skip_special_tokens=False,
        # random_seed=config.training.seed,  # eval模式使用固定seed，不使用epoch
        # stop_words=["<|endoftext|>"],
    )

    print("expect:", gen_config)
    start_with_think = config.rollout.start_with_think
    # dataset_name = config.dataset.train_dataset

    outputs_name = "eval-" + model_path.replace("/", ".") + "-" + dataset_name + "-" + str(epoch)
    # Load data
    print("Loading data...")
    if accelerator.is_main_process:
        # rank0 负责构建数据集并写缓存
        dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")

    # 等 rank0 处理完再继续
    accelerator.wait_for_everyone()
    # 所有 rank 都能安全读取缓存
    if not accelerator.is_main_process:
        dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")

    ds = dataset['train']
    prompts = ds['question']
    gts = ds['ground_truth_answer']

    reason_prompt = "<|im_start|>user\n{problem}\nPlease reason step by step, and put your final answer within $\\boxed{{}}$.<|im_end|>\n<|im_start|>assistant\n"

    # reason_prompt = "<|im_start|>user\n{problem}\nPlease reason step by step in <think>...</think>, and put your final answer within $\\boxed{{}}$ in <answer>...</answer>.<|im_end|>\n<|im_start|>assistant\n"
    # start_with_think=True
    if start_with_think:
        reason_prompt = reason_prompt + "<think>"
        
    # 构建带原始索引的数据
    all_data = list(enumerate(zip(prompts, gts))) # [(idx, (prompt, gt)), ...]
    # random.seed(config.training.seed+epoch)
    # random.seed(config.training.seed)
    # Directly sample and shuffle the original data
    num_tasks_total = len(all_data)

    rank = accelerator.process_index
    num_processes = accelerator.num_processes
    local_data = get_data_chunk(all_data, accelerator.num_processes, accelerator.process_index)

    if len(local_data) == 0:
        accelerator.print(f"[Rank {rank}] No prompts assigned.")
        # 仍需返回空结果以便 gather
        data = []
        count_low = 0
        count_mid = 0
        count_high = 0
    else:
        indices, prompt_gt_pairs = zip(*local_data)
        local_prompts, local_gts = zip(*prompt_gt_pairs)

        # Apply chat template
        prompts_list = [reason_prompt.format(problem=p) for p in local_prompts]
        
        print("参考prompt: ", repr(prompts_list[0]))

        # sample函数现在返回完整的data列表，支持k次采样
        k_sample = getattr(config.evaluation, 'num_response_per_task', 1)  # 从配置中获取k值

        outputs = sample(model_path, backend_config, gen_config, prompts_list, k_sample, config, accelerator)

        # Debug: check k-sample consistency
        expected_outputs = len(prompts_list) * k_sample
        actual_outputs = len(outputs)
        accelerator.print(f"[Rank {rank}] k_sample={k_sample} prompts={len(prompts_list)} expected_outputs={expected_outputs} actual_outputs={actual_outputs}")
        used_pairs = actual_outputs // k_sample
        leftover = actual_outputs - used_pairs * k_sample
        if leftover != 0:
            accelerator.print(f"[Rank {rank}] Warning: leftover outputs not forming full k groups: leftover={leftover}")

        data = []
        # 统计correctness分布
        count_low = 0  # <0.2
        count_mid = 0  # 0.2-0.8
        count_high = 0  # >0.8
        
        for original_idx in range(0, len(outputs) // k_sample):
            question = local_prompts[original_idx]
            prompt = prompts_list[original_idx]
            gt = local_gts[original_idx]

            combined_texts = []
            combined_step_maps = []
            extracted_answers = []
            response_lengths = []
            correctness_list = []
            raw_rewards = []
            speed_rewards = []
            for i in range(k_sample):
                output_idx = original_idx * k_sample + i
                o = outputs[output_idx]
                text = o.text
                token_ids = o.token_ids
                step_map = o.step_map

                combined_texts.append(text)
                combined_step_maps.append(step_map)
                extracted_answers.append(extract_final_boxed_answer(text))
                response_lengths.append(len(token_ids))

                if reward == "math":
                    correctness = math_judge([text], [gt])[0]
                    if correctness == 0:
                        if "boxed" in text:
                            correct_reward = 0.1
                        else:
                            correct_reward = 0
                        speed_reward = 0.0
                    else:
                        correct_reward = 1
                        if config.rollout.speed:
                            speed_reward = len(step_map)/len(set(step_map))/config.rollout.block_size
                            correct_reward += speed_reward
                        else:
                            speed_reward = 0.0
                        
                    # cosine_reward_score = cosine_reward([text], [gt], response_token_ids=[token_ids])[0]
                    # reward_score = cosine_reward_score
                    # format_reward = 1 if "boxed" in text and not correctness else -1.0
                    # reward_score = min(format_reward, cosine_reward_score) 
                    # repetition_reward_score = repetition_reward([text])[0]
                    # format_reward_score = format_reward([text])[0]
                    # correctness = 1 if cosine_reward_score > 0 else 0
                    correctness_list.append(correctness)
                    raw_rewards.append([correct_reward])
                    speed_rewards.append(speed_reward)
                else:
                    print(text,gt)
                    correctness_list.append(0)
                    speed_rewards.append(0.0)

            # 计算mean_correctness并统计分布
            mean_correctness = sum(correctness_list) / len(correctness_list)
            if mean_correctness < 0.2:
                count_low += 1
            elif mean_correctness <= 0.8:
                count_mid += 1
            else:
                count_high += 1

            raw_rewards = [sum(x) for x in raw_rewards]

            rewards = raw_rewards

            data.append({
                "question": [question]*k_sample,
                "prompt": [prompt]*k_sample,
                "response": combined_texts,
                "ground_truth_answer": [gt]*k_sample,
                "full_output": combined_texts,
                "extracted_output": extracted_answers,
                "response_length": response_lengths,
                "correctness": correctness_list,
                "rewards": rewards,
                "raw_rewards": raw_rewards,
                "speed_rewards": speed_rewards,
                "step_map": combined_step_maps,
            })

        accelerator.print(f"[Rank {rank}] Correctness distribution: <0.2: {count_low}, 0.2-0.8: {count_mid}, >0.8: {count_high}")
        # Debug: built data k consistency
        if len(data) > 0:
            ks = set(len(d['prompt']) for d in data)
            flattened = sum(len(d['prompt']) for d in data)
            accelerator.print(f"[Rank {rank}] Built tasks={len(data)} k_set={sorted(list(ks))} flattened={flattened}")
        
    accelerator.wait_for_everyone()

    # Gather局部统计
    all_counts = gather_object([count_low, count_mid, count_high])
    
    # 直接gather data
    all_data = gather_object(data)
    
    # 在主进程打印全局统计并保存结果
    if accelerator.is_main_process:
        # 展平多 rank 数据
        if isinstance(all_data[0], list):
            all_data_flattened = [item for rank_data in all_data for item in rank_data]
        else:
            all_data_flattened = all_data
        
        total_low = sum(all_counts[i] for i in range(0, len(all_counts), 3))
        total_mid = sum(all_counts[i] for i in range(1, len(all_counts), 3))
        total_high = sum(all_counts[i] for i in range(2, len(all_counts), 3))
        total = total_low + total_mid + total_high
        print(f"\n{'='*60}")
        print(f"Global Correctness Distribution Statistics")
        print(f"Epoch: {epoch} | Sampled Tasks: {num_tasks_total} | Collected: {total}")
        print(f"  <0.2:     {total_low:5d} ({total_low/total*100:5.1f}%)")
        print(f"  0.2-0.8:  {total_mid:5d} ({total_mid/total*100:5.1f}%)")
        print(f"  >0.8:     {total_high:5d} ({total_high/total*100:5.1f}%)")
        print(f"{'='*60}\n")
        
        # 保存结果到 JSON 文件
        import json
        output_file_name = project_name + "/temp_data/outputs-" + outputs_name + ".json"
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(all_data_flattened, f, indent=2, ensure_ascii=False)
        print(f"{output_file_name} 保存成功，一共 {len(all_data_flattened)} 条数据")
    
    return all_data

