import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed


from models import SDARForCausalLM
from transformers import AutoTokenizer
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader

SYSTEM_PROMPT_LEN = 28

from train.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx]
        )


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = config.model.pretrained_model

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)


    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    # Resume from checkpoint by global_step
    resume_from_step = getattr(config.experiment, "resume_from_step", None)
    if resume_from_step is not None and resume_from_step > 0:
        model_path = config.experiment.prefix_dir + project_name + f"/ckpt/checkpoint-step-{resume_from_step}"
        logger.info(f"Resuming from checkpoint-step-{resume_from_step}")
    elif config.experiment.current_epoch != 1:
        model_path = config.experiment.prefix_dir + project_name + f"/ckpt/checkpoint-epoch-{config.experiment.current_epoch-1}"
    else:
        model_path = pretrained_model

    pretrained_model = model_path

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
    # model = SDARForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    # calculate loss ourselves, needs logits，so aviod fuse CE
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False  
    

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")




    def collapse_k_unique(lst, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))

        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]
    
    


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")


    def simple_collate(batch):
        extended_input_ids, p_mask, tok_idx_ext, labels = zip(*batch)     # Tensor(L)
        return {
            "extended_input_ids":  torch.stack(extended_input_ids),
            "p_mask":  torch.stack(p_mask),
            "tok_idx_ext":  torch.stack(tok_idx_ext),
            "labels":  torch.stack(labels)
        }
    


    
    with open("./data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)
    #dataset_load = dataset_load[10000:20000]

    post_num = config.training.post_num
    large_batch_size = config.training.get("large_batch_size", 512)
    if large_batch_size == -1:
        large_batch_size = len(dataset_load)

    
    
    def make_basic_block_attention(
        N: int,
        start_pos: int,            # = L0
        block_size: int,           # = b
    ) -> torch.Tensor:
        B = 1
        L0     = start_pos
        L1     = (N - L0) // 2          # N = L0 + 2·L1 
        assert L0 + 2 * L1 == N, "input length must be L0 + 2*L1"

        # all -inf first
        bias = torch.full((B, 1, N, N), 0)


        rows = torch.arange(L0 + L1, L0 + 2 * L1)              # (L1,)
        rows_token = torch.arange(L0, L0 + L1)              # (L1,)

        # update block by block
        for bi in range((L1 + block_size - 1) // block_size):
            #  [bi*b , min((bi+1)*b, L1))
            left_end   = L0 + min((bi) * block_size, L1)        
            right_start= L0 + L1 + (left_end - L0)

            i_start = bi * block_size
            i_end   = min((bi + 1) * block_size, L1)              # no i_end

            block_rows = rows[i_start:i_end]                    
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
            bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1

            block_rows = rows_token[i_start:i_end]
            left_end   = L0 + min((bi + 1) * block_size, L1)
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
        
        if L0 > 0:
            num_blocks_pre = (L0 + block_size - 1) // block_size
            for bi in range(num_blocks_pre):
                # row interval [row_start, row_end)
                row_end   = max(L0 - bi * block_size, 0)
                row_start = max(L0 - (bi + 1) * block_size, 0)
                if row_end > row_start:
                    block_rows = torch.arange(row_start, row_end)
                    bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
        
        return bias        # (B,1,N,N)
    
    
    

    def process_pad(attn, input_ids, start_pos, L0, L1):
        N = L0 + 2 * L1
        device = input_ids.device

        cols = torch.arange(N, device=device)                  # (N,)
        key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)  # (B, N)

        # set -inf
        attn.masked_fill_(key_mask[:, None, None, :], 0)

        # aviod +-inf or none in forward
        A = attn[:, 0]  # (B, N, N)
        bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
        b, r = bad.nonzero(as_tuple=True)
        A[b, r, :] = 0; A[b, r, r] = 1  

        return attn





    def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
        """
        Perform a single "round" on one sample b:
        - For each block, take the minimum non -1 value in step_map.
        - Create pmask (positions equal to the block minimum).
        - Create a noise mask for the extended segment (positions >= block minimum).
        - Mark the chosen minimum positions in step_map as -1 for the next round.

        Returns:
        extended_input_ids_b : Tensor with duplicated + masked response segment
        pmask_b              : Boolean mask for tokens selected in this round
        new_step_map_b       : Updated step_map (selected positions set to -1)
        has_any              : Whether any position was selected in this round
        """
        device = input_ids_b.device
        NB = (L1 + block_size - 1) // block_size
        pad_len = NB * block_size - L1

        # Reshape step_map into [NB, block_size], fill last incomplete block with -1
        step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
        step_pad[:L1] = step_map_b
        step_blk = step_pad.view(NB, block_size)                      # [NB, Bk]

        valid = step_blk.ge(0)                                        # Valid positions (not -1)
        big = torch.iinfo(step_blk.dtype).max
        tmp = step_blk.masked_fill(~valid, big)                       # Fill invalid positions with a large value
        min_vals, _ = tmp.min(dim=1, keepdim=True)                    # Current minimum for each block

        # Select positions equal to block minimum (only valid positions)
        pmask_blk = step_blk.eq(min_vals) & valid                     
        if not pmask_blk.any():
            # No positions left to select in this round
            return None, None, step_map_b, False

        # Noise mask for extended segment: mark positions >= block minimum
        ge_mask_blk = step_blk.ge(min_vals) & valid                   # [NB, Bk]

        # Flatten back to length L1 (discard padding)
        pmask_tail = pmask_blk.view(-1)[:L1]                          # [L1]
        ge_mask_tail = ge_mask_blk.view(-1)[:L1]                      # [L1]

        # Construct pmask_b: [0:L0] = False, [L0:] = pmask_tail
        pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
        pmask_b[L0:] = pmask_tail

        # Build extended segment: duplicate response and replace noise positions with mask_id
        tail = input_ids_b[L0:L0+L1].clone()
        tail[ge_mask_tail] = mask_id

        
        extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
        extended_input_ids_b[:L0+L1] = input_ids_b
        extended_input_ids_b[L0+L1:] = tail

        # Update step_map: mark selected minimum positions as -1 for the next round
        new_step_map_b = step_map_b.clone()
        new_step_map_b[pmask_tail] = -1

        return extended_input_ids_b, pmask_b, new_step_map_b, True




    def collect_training_data(input_ids, step_map_list):

        B, L = input_ids.shape
        L0    = start_pos
        L1    = L - L0
        block_size = config.training.block_size

        lower = config.training.lower_p
        upper = config.training.upper_p


        '''
        if config.training.method == "semi-ar":

            extended_input_ids_list, pmask_list = [], []
            
            
            for b in range(B):

                extended_input_ids_b = input_ids[b]
                pmask_b = torch.zeros(start_pos, dtype=torch.bool)
                
                for j in range(int((L1 - 1) / block_size) + 1):

                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)

                    pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                    #pmask_b_j = torch.rand(end - start) <= torch.linspace(lower, upper, steps=end - start)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)

                    noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)

                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)
                
                extended_input_ids_list.append(extended_input_ids_b)
                pmask_list.append(pmask_b)'''
        
        if config.training.method == "semi-ar":

            extended_input_ids_list, pmask_list = [], []

            # Pre-compute the global linear probability
            

            for b in range(B):

                prob_ramp = torch.empty(L1).uniform_(lower, upper)
                
                # 1) Sample random numbers once for each position in the tail segment of this sample
                rand_tail = torch.rand(L1)
                pmask_tail = rand_tail <= prob_ramp  # [L1], True means to mask

                # 2) Construct the pmask for this sample (prefix all False, tail uses pmask_tail)
                pmask_b = torch.cat([
                    torch.zeros(L0, dtype=torch.bool),
                    pmask_tail
                ], dim=0)  # [L]

                # 3) Construct the noisy tail and append it to the original sequence to get the extended sequence
                noise_tail = input_ids[b, L0:].clone()            # [L1]
                noise_tail.masked_fill_(pmask_tail, mask_id)      # Replace masked positions
                extended_b = torch.cat([input_ids[b], noise_tail], dim=0)  # [L + L1]

                extended_input_ids_list.append(extended_b)
                pmask_list.append(pmask_b)

        elif config.training.method == "trace":

            for b in range(B):
                step_map_i = step_map_list[b]

                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)
            
            step_map = torch.as_tensor(step_map_list, dtype=torch.long)

            assert step_map.shape[1] == L1

            extended_input_ids_list, pmask_list = [], []

            for b in range(B):
                step_b = step_map[b]
                while True:
                    out = one_round_vectorized(
                        input_ids_b=input_ids[b],
                        step_map_b=step_b,
                        L0=L0,
                        L1=L1,
                        block_size=block_size,
                        mask_id=mask_id,
                    )
                    extended_b, pmask_b, step_b, has_any = out
                    if not has_any:
                        break

                    extended_input_ids_list.append(extended_b)
                    pmask_list.append(pmask_b)

        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask =  torch.stack(pmask_list, dim=0).to(torch.bool)
        
        pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask        
        if post_num is not None:
            cum_pad = torch.cumsum(pad_resp.int(), dim=1)
            p_mask &= ~(pad_resp & (cum_pad > post_num))
        
        labels = extended_input_ids[:, :L].clone()

        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)      
        tok_idx = valid.long().cumsum(dim=-1) - 1         
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_resp = tok_idx[:, start_pos:]  
        tok_idx_ext  = torch.cat([tok_idx, tok_idx_resp], dim=1)

        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)

        extended_input_ids = extended_input_ids[keep]
        p_mask            = p_mask[keep]
        tok_idx_ext       = tok_idx_ext[keep]
        labels            = labels[keep]

        assert input_ids.shape[0] == extended_input_ids.shape[0] if config.training.shrink==4 else True, "shrink 4: input_ids.shape[0] != extended_input_ids.shape[0]"

        return extended_input_ids, p_mask, tok_idx_ext, labels
        

    
    # Estimate training schedule parameters (assuming 4x expansion)
    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    estimated_dataset_size = len(dataset_load) * 4  # Assume 4x expansion
    num_update_steps_per_epoch = math.ceil(estimated_dataset_size / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )


    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    
    # Create a dummy dataloader for DeepSpeed to infer batch_size
    dummy_data = [(0, 0, 0, 0)] * (config.training.batch_size_lm * accelerator.num_processes)
    dummy_dataloader = DataLoader(
        dummy_data,
        batch_size=config.training.batch_size_lm,
    )
    
    model, optimizer, lr_scheduler, dummy_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dummy_dataloader
    )



    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Estimated dataset size per epoch = {estimated_dataset_size} (4x expansion)")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    
    # Log micro_batch_size if configured
    micro_batch_size_config = getattr(config.training, 'micro_batch_size', None)
    if micro_batch_size_config is not None:
        logger.info(f"  Micro batch size for forward pass = {micro_batch_size_config} (memory optimization enabled)")
    else:
        logger.info(f"  Micro batch size = None (full batch forward pass)")


    first_epoch = config.experiment.current_epoch
    
    # Resume 参数：resume_from_step 用于加载 checkpoint，resume_batch_count 用于跳过 large_batch
    resume_from_step = getattr(config.experiment, "resume_from_step", None)
    resume_batch_count = getattr(config.experiment, "resume_batch_count", 0)
    
    if resume_from_step is not None and resume_from_step > 0:
        if accelerator.is_main_process:
            logger.info(
                f"Resume from global_step={resume_from_step}, "
                f"will skip first {resume_batch_count} large_batches."
            )
        large_batches_to_skip = resume_batch_count
    else:
        large_batches_to_skip = 0

    data_time_m = AverageMeter()
    end = time.time()

    import torch.nn.functional as F

    

    ''' To be updated
    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels):

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)

        labels_trim = labels[:, L0:].clone()                             # (B, L1)
        resp_keep   = p_mask[:, L0:]                                        # (B, L1) True=训练
        labels_trim[~resp_keep] = -100   

        out = model(
            input_ids      = extended_input_ids,
            attention_mask = attention_mask,
            position_ids   = tok_idx_ext,
            labels         = labels_trim,    # 注意：这里与 logits_to_keep 对齐
            logits_to_keep = L1,             # 关键：只算最后 L1 个 time steps
            return_dict    = True,
            use_cache      = False,
        )

        loss_lm = out.loss


        tokens_kept = (labels_trim != -100).sum()
        if tokens_kept > 0:
            loss_lm = loss_lm * (tokens_kept.to(loss_lm.dtype) / B)

        
        return loss_lm
    '''
    

    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels, basic_block_attention, start_pos, micro_batch_size=None):
        """
        Forward process with optional micro-batching to reduce memory usage.
        
        Args:
            extended_input_ids: Input token IDs
            p_mask: Mask for tokens to train
            tok_idx_ext: Position IDs
            labels: Target labels
            basic_block_attention: Attention mask
            start_pos: Start position
            micro_batch_size: If specified, split batch into smaller chunks for forward pass
        """
        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        # If micro_batch_size is not specified or >= B, process all at once (original behavior)
        if micro_batch_size is None or micro_batch_size >= B:
            attention_mask = basic_block_attention.clone()
            attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
            attention_mask = process_pad(attention_mask, extended_input_ids, start_pos, L0, L1)

            logits = model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext).logits
            logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

            log_probs = F.log_softmax(logits, dim=-1)
            logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)     # (B, T)
            num_mask = torch.clamp(p_mask.sum(dim=1), min=1)
            loss_lm = - (logp_tok * p_mask).sum(dim=1) / num_mask
            loss_lm = loss_lm.sum() / B
            
            return loss_lm
        
        # Process in micro-batches
        all_logp_tok = []
        all_num_mask = []
        
        for i in range(0, B, micro_batch_size):
            end_idx = min(i + micro_batch_size, B)
            micro_B = end_idx - i
            
            # Slice inputs for this micro-batch
            micro_extended_input_ids = extended_input_ids[i:end_idx]
            micro_p_mask = p_mask[i:end_idx]
            micro_tok_idx_ext = tok_idx_ext[i:end_idx]
            micro_labels = labels[i:end_idx]
            
            # Process micro-batch
            micro_attention_mask = basic_block_attention.clone()
            micro_attention_mask = micro_attention_mask.repeat_interleave(micro_B, dim=0).to(device)
            micro_attention_mask = process_pad(micro_attention_mask, micro_extended_input_ids, start_pos, L0, L1)
            
            micro_logits = model(
                input_ids = micro_extended_input_ids, 
                attention_mask = micro_attention_mask, 
                position_ids = micro_tok_idx_ext
            ).logits
            micro_logits = torch.cat([micro_logits[:, :L0, :], micro_logits[:, L0 + L1 :, :]], dim=1)
            
            micro_log_probs = F.log_softmax(micro_logits, dim=-1)
            micro_logp_tok = micro_log_probs.gather(dim=-1, index=micro_labels.unsqueeze(-1)).squeeze(-1)
            micro_num_mask = torch.clamp(micro_p_mask.sum(dim=1), min=1)
            
            all_logp_tok.append(micro_logp_tok)
            all_num_mask.append(micro_num_mask)
        
        # Concatenate results from all micro-batches
        logp_tok = torch.cat(all_logp_tok, dim=0)  # (B, T)
        num_mask = torch.cat(all_num_mask, dim=0)  # (B,)
        
        # Calculate loss
        loss_lm = - (logp_tok * p_mask).sum(dim=1) / num_mask
        loss_lm = loss_lm.sum() / B
        
        return loss_lm






    from tqdm.auto import tqdm

    # 初始化 global_step: 如果从 checkpoint resume，则从 resume_from_step 开始
    if resume_from_step is not None and resume_from_step > 0:
        global_step = resume_from_step
        if accelerator.is_main_process:
            logger.info(f"Starting from global_step={global_step}")
    else:
        global_step = 0
    
    prev_num_masks = None  # Store previous epoch's mask count for comparison
    total_large_batches_seen = 0  # 全局已经遍历的 large_batch 总数（跨 epoch）
    
    for epoch in range(first_epoch, num_train_epochs):

        logger.info(f"Epoch {epoch}/{num_train_epochs}")
        
        # If passed along, set the training seed now.
        if config.training.seed is not None:
            set_seed(config.training.seed + epoch)

        model.train()
        
        # 将dataset_load切分成多个大batch
        num_large_batches = math.ceil(len(dataset_load) / large_batch_size)
        logger.info(f"Num large batches: {num_large_batches}")
        
        epoch_step = 0  # 本 epoch 内实际执行的训练 step 数
        
        for large_batch_idx in range(num_large_batches):
            total_large_batches_seen += 1
            
            # Resume 逻辑：跳过已经训练过的 large_batch
            if total_large_batches_seen <= large_batches_to_skip:
                if accelerator.is_main_process and total_large_batches_seen == large_batches_to_skip:
                    logger.info(
                        f"[Resume] Skipped {large_batches_to_skip} large_batches, "
                        f"now starting training from large_batch {total_large_batches_seen + 1} "
                        f"(epoch={epoch}, large_batch_idx={large_batch_idx+1})"
                    )
                continue
            
            start_idx = large_batch_idx * large_batch_size
            end_idx = min((large_batch_idx + 1) * large_batch_size, len(dataset_load))
            dataset_batch = dataset_load[start_idx:end_idx]
            
            logger.info(f"Processing large batch {large_batch_idx+1}/{num_large_batches} (samples {start_idx}:{end_idx})")
            
            # 对当前大batch进行uni_prompting
            prompt_list = []
            response_list = []
            step_map_list = []
            for x in dataset_batch:
                prompt_list.append(x["prompt"])
                response_list.append(x["response"])
            
            input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))
            _, L = input_ids_lm.shape
            L0 = start_pos
            L1 = L - L0
            
            if accelerator.is_main_process:
                logger.info(f"  Batch {large_batch_idx+1}: L0={L0}, L1={L1}")
            
            for x in dataset_batch:
                if "step_map" not in x.keys():
                    step_map_list.append([j for j in range(L1)])
                else:
                    step_map_i = x["step_map"]
                    if len(step_map_i) > L1:
                        step_map_i = step_map_i[:L1]
                    else:
                        step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
                    step_map_list.append(step_map_i)
            
            # 创建basic_block_attention
            basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size)
            basic_block_attention = basic_block_attention.cpu()
            
            # 收集训练数据
            extended_input_ids, p_mask, tok_idx_ext, labels = collect_training_data(input_ids_lm, step_map_list)
            
            dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels)
            
            train_dataloader_lm = DataLoader(
                dataset_lm,
                batch_size=config.training.batch_size_lm,
                sampler=None,
                collate_fn=simple_collate,
                num_workers=0
            )
            
            train_dataloader_lm = accelerator.prepare(train_dataloader_lm)
            
            logger.info(f"  Large batch {large_batch_idx+1} - Num training data = {extended_input_ids.shape[0]}")
            
            progress_bar = tqdm(
                train_dataloader_lm,
                desc=f"Epoch {epoch} LargeBatch {large_batch_idx+1}/{num_large_batches}",
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True,         
                leave=True                
            )

            for batch_idx, batch in enumerate(progress_bar):

                epoch_step += 1
                data_time_m.update(time.time() - end)


                extended_input_ids_batch = batch["extended_input_ids"].to(accelerator.device)

                p_mask_batch = batch["p_mask"].to(accelerator.device)
                tok_idx_ext_batch = batch["tok_idx_ext"].to(accelerator.device)
                labels_batch = batch["labels"].to(accelerator.device)

                # Use micro_batch_size if configured to reduce memory usage
                micro_batch_size = getattr(config.training, 'micro_batch_size', None)
                
                loss_lm = forward_process(
                        extended_input_ids=extended_input_ids_batch,
                        p_mask=p_mask_batch,
                        tok_idx_ext=tok_idx_ext_batch,
                        labels=labels_batch,
                        basic_block_attention=basic_block_attention,
                        start_pos=start_pos,
                        micro_batch_size=micro_batch_size
                    )
                
                if accelerator.is_main_process:
                    accelerator.print(f"Step {epoch_step}, loss={loss_lm.item():.4f}")
                
                # 清理batch输入数据以节省显存
                del extended_input_ids_batch, p_mask_batch, tok_idx_ext_batch, labels_batch
                    
                accelerator.backward(loss_lm)
                
                # 每个batch后同步清理显存，避免cache flush
                accelerator.wait_for_everyone()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if epoch_step % accelerator.gradient_accumulation_steps == 0:
                    if config.training.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(),
                                                    config.training.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    
                    if accelerator.is_main_process:   
                        accelerator.print(f"global_step={global_step}, loss={loss_lm.item():.4f}")
                        accelerator.log(
                            {
                                "train/loss": loss_lm.item(),
                                "train/lr": lr_scheduler.get_last_lr()[0],
                                "train/global_step": global_step,
                                "train/epoch": epoch,
                            },
                            step=global_step,
                        )
                    
                    # 清理梯度更新后的显存，所有rank同步
                    accelerator.wait_for_everyone()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if global_step % config.experiment.save_steps == 0:
                        save_name = f"checkpoint-step-{global_step}"
                        accelerator.print(f"[RANK {accelerator.process_index}] Triggering save at global_step={global_step}")
                        save_checkpoint(model, tokenizer, config, accelerator, save_name)
            
            # 当前大batch训练完成，清理内存，所有rank同步
            del extended_input_ids, p_mask, tok_idx_ext, labels, dataset_lm, train_dataloader_lm
            del input_ids_lm, prompt_list, response_list, step_map_list, basic_block_attention, dataset_batch
            accelerator.wait_for_everyone()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # epoch结束后检查是否还有没更新的梯度
        if epoch_step % accelerator.gradient_accumulation_steps != 0:
            if config.training.max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(),
                                            config.training.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        save_name = f"checkpoint-epoch-{epoch}"
        accelerator.print(f"[RANK {accelerator.process_index}] Triggering save at epoch={epoch}")
        save_checkpoint(model, tokenizer, config, accelerator, save_name)  # ← 所有进程都调用

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)

    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, name):

    from pathlib import Path
    import time, json, shutil, os, glob

    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        save_dir = save_base / name

        # 直接从 model_to_save 抽取权重，避免 get_state_dict 丢参数
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(save_dir)

        # 保存当前 config，而不是覆盖旧的
        model_to_save.config.save_pretrained(save_dir)

        # 复制自定义源码文件
        model_path = config.model.pretrained_model
        for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
            for fn in glob.glob(os.path.join(model_path, pattern)):
                dst = os.path.join(save_dir, os.path.basename(fn))
                if not os.path.exists(dst):
                    shutil.copy2(fn, dst)

        # 记录元数据
        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": getattr(config, "global_step", None)
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved checkpoint to {save_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()