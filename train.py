# LINT_ME
import argparse
import os
import logging
import shutil
import copy
import random

import numpy as np
import transformers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import diffusers
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from fudoki.model import instantiate_model
from fudoki.janus.models import VLChatProcessor
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.data.navsim import SupervisedDataset
from flow_matching.utils.flow import get_source_distribution


logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def init_numeric_and_special_tokens(
    model,
    tokenizer,
    numeric_tokens,
    noise_scale: float = 0.01,
):
    emb = model.get_input_embeddings().weight
    device = emb.device
    dim = emb.shape[1]

    def tok_ids_for_text(text: str):
        # Get subword ids (no specials), filter out -100/None if any
        ids = tokenizer.encode(text,add_special_tokens=False)
        return [i for i in ids if isinstance(i, int) and i >= 0]

    # ---- Build a numeric "base" vector from digits and dot ----
    digit_ids = []
    for d in "0123456789":
        _ids = tok_ids_for_text(d)
        digit_ids.extend(_ids)
    dot_ids = tok_ids_for_text(".")

    base_chunks = []
    if digit_ids:
        base_chunks.append(emb[torch.tensor(digit_ids, device=device)].mean(dim=0))
    if dot_ids:
        base_chunks.append(emb[torch.tensor(dot_ids, device=device)].mean(dim=0))
    if base_chunks:
        numeric_base = torch.stack(base_chunks, dim=0).mean(dim=0)
    else:
        # fallback if tokenizer lacks digits/dot as standalone pieces
        numeric_base = torch.zeros(dim, device=device)

    # ---- Initialize numeric tokens ----
    for t in numeric_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid < 0:
            continue
        noise = noise_scale * torch.randn(dim, device=device)
        emb[tid] = numeric_base + noise


def training_step(
    model,
    x_1,
    source_distribution,
    data_info,
    path,
    time_epsilon = 0.001,
    loss_fn = CrossEntropyLoss(),
    stage="s1",
    vl_chat_processor=None,
    args=None,
):
    x_0 = source_distribution.sample_like(x_1)
    t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)

    if stage == "s1":
        x_t = x_1
    elif stage == "s2":
        # update emb layer when using num tokenizer
        path_sample = path.sample(x_0, x_1, t)
        x_t = path_sample.x_t

    # text_token_mask==1 ==> generated text token
    x_t = x_t * data_info['text_token_mask'] + x_1 * (1 - data_info['text_token_mask'])
    data_info['understanding_img'] = data_info['understanding_img'].to(dtype=model.dtype)

    _, txt_logits = model(x_t, data_info)

    b, _, c = txt_logits.shape
    mask = data_info['text_token_mask'].unsqueeze(-1).bool()
    txt_logits = txt_logits.masked_select(mask)
    txt_logits = txt_logits.view(b, -1, c)
    x_1 = x_1.masked_select(mask.squeeze(-1)).view(b, -1)

    loss = ce_loss = loss_fn(txt_logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
    loss_dict = {"ce_loss": ce_loss.detach().item()}

    if stage == "s2":
        start_mask = x_1 >= vl_chat_processor.num_start_id
        end_mask = x_1 <= vl_chat_processor.num_end_id - 1
        action_mask = start_mask & end_mask

        if action_mask.any():
            if args.l2_loss_weight > 0:
                pred_probabilities = F.softmax(txt_logits, dim=-1)
                pred_ids = torch.argmax(pred_probabilities, dim=-1)
                pred_num_ids = pred_ids.masked_select(action_mask)
                pred_nums = vl_chat_processor.min_num + (pred_num_ids - vl_chat_processor.num_start_id) * vl_chat_processor.interval
                pred_nums = torch.clip(pred_nums, vl_chat_processor.min_num, vl_chat_processor.max_num)

                tgt_num_ids = x_1.masked_select(action_mask) # [N]
                tgt_nums = vl_chat_processor.min_num + (tgt_num_ids - vl_chat_processor.num_start_id) * vl_chat_processor.interval

                l2_loss = args.l2_loss_weight + F.mse_loss(pred_nums, tgt_nums, reduction="mean")
                loss = loss + l2_loss
                loss_dict["l2_loss"] = l2_loss.detach().item()

    loss_dict["loss"] = loss.detach().item()
    return loss, loss_dict


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_grad_batches,
        log_with="tensorboard",
        project_dir=args.output_dir
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # random seed
    seed = args.seed + accelerator.process_index
    if args.random_seed:
        seed = seed + random.randint(0, 500)
    set_seed(seed)
    logger.info(f"accelerator.process_index: {accelerator.process_index},   seed: {seed} \n")

    # work dir
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.yaml")
        OmegaConf.save(args, config_path)
        accelerate_config_path = os.path.join(args.output_dir, "accelerate_config_ds2.yaml")
        shutil.copyfile(
            "./config/accelerate_config_ds2.yaml",
            accelerate_config_path
        )
    accelerator.wait_for_everyone()

    # dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # prepare dataset
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    if args.use_quantize:
        origin_len = len(vl_chat_processor.tokenizer)
        num_tokens = [f"{x:.2f}" for x in np.linspace(-100, 100, 20001)]
        num_tokens_length = len(num_tokens)
        vl_chat_processor.tokenizer.add_tokens(num_tokens)

        vl_chat_processor.num_start_id = origin_len
        vl_chat_processor.num_end_id = origin_len + num_tokens_length - 1
        vl_chat_processor.min_num = -100
        vl_chat_processor.max_num = 100
        vl_chat_processor.interval = 0.01

        logger.info(f"Total number tokens: {num_tokens_length}")

    # data
    dataset = SupervisedDataset(
        data_list=args.data_list,
        vl_chat_processor=vl_chat_processor,
        txt_max_length=args.txt_max_length
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    logger.info(f"Max txt length: {args.txt_max_length}")
    logger.info(f"Total data samples: {len(dataset)}")

    # prepare model
    stage = args.stage
    logger.info(f"Training stege: {stage}")
    model = instantiate_model(
        args.pretrain_model_path
    ).to(weight_dtype)
    model.uncond_prob = args.uncond_prob

    if os.path.exists(args.pretrain_path):
        sd = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(sd, strict=True)
        model = model.to(weight_dtype)
        logger.info(f"Loading pretrain ckpt from {args.pretrain_path}")

    if stage == "s1":
        model.language_model.resize_token_embeddings(args.vocab_size + num_tokens_length)
        init_numeric_and_special_tokens(
            model.language_model,
            vl_chat_processor.tokenizer,
            numeric_tokens=num_tokens
        )
    elif stage == "s2":
        if os.path.exists(args.new_embedding_path):
            old_emb = copy.deepcopy(model.language_model.get_input_embeddings())

            with torch.serialization.safe_globals([torch.nn.modules.sparse.Embedding]):
                new_emb_state = torch.load(args.new_embedding_path, map_location="cpu")
                logger.info(f"Loading new embedding from {args.new_embedding_path}")

            if isinstance(new_emb_state, dict) and "weight" in new_emb_state:
                weight = new_emb_state["weight"]
                new_emb = torch.nn.Embedding(weight.size(0), weight.size(1))
                new_emb.load_state_dict(new_emb_state)
            else:
                new_emb = new_emb_state

            model.language_model.resize_token_embeddings(args.vocab_size + num_tokens_length)

            # origin_len = old_emb.weight.shape[0]
            origin_len = vl_chat_processor.num_start_id
            new_emb.weight.data[:origin_len, :] = old_emb.weight.data[:origin_len, :]

            model.language_model.set_input_embeddings(new_emb)

        if os.path.exists(args.ckpt_path):
            if model.language_model.model.embed_tokens.weight.shape[0] != args.vocab_size + num_tokens_length:
                model.language_model.resize_token_embeddings(args.vocab_size + num_tokens_length)
            sd = torch.load(args.ckpt_path, map_location='cpu')
            model.load_state_dict(sd, strict=True)
            model = model.to(weight_dtype)
            logger.info(f"Loading ckpt from {args.ckpt_path}")

    # prepare path
    path = MixtureDiscreteSoftmaxProbPath(
        mode='text',
        embedding_path=args.text_embedding_path
    )
    if args.use_quantize:
        path.set_embedding(model.language_model.get_input_embeddings())
    else:
        logger.info("No quantize!")

    logger.info(f"path.a = {path.a}")
    logger.info(f"path.c = {path.c}")

    # set trainable params
    model.requires_grad_(False)
    if stage == "s1":
        model.language_model.requires_grad_(False)
        model.language_model.model.embed_tokens.requires_grad_(True)
        model.language_model.lm_head.requires_grad_(True)
    elif stage == "s2":
        model.language_model.requires_grad_(True)
        if args.train_llm_emb:
            model.language_model.model.embed_tokens.requires_grad_(True)
        else:
            model.language_model.model.embed_tokens.requires_grad_(False)

    trainable_params = list(
        filter(lambda p: p.requires_grad, model.parameters())
    )

    # log trainable params
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable params: {name}")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_trainable/1e9:.3} B")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    # lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes
    )

    # accelerator
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    source_distribution = get_source_distribution(
        source_distribution=args.source_distribution,
        vocab_size=args.vocab_size + num_tokens_length if args.use_quantize else args.vocab_size,
    )


    global_step = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # training loop
    for epoch in range(args.max_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.max_epochs}")
        logger.info(f"training sample length: {len(dataloader)}")

        for _, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                x_1 = batch["input_ids"].to(dtype=torch.long)
                loss, logs = training_step(
                    x_1=x_1,
                    model=model,
                    source_distribution=source_distribution,
                    data_info=batch,
                    path=path,
                    stage=stage,
                    vl_chat_processor=vl_chat_processor,
                    args=args
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs["lr"] = optimizer.param_groups[0]['lr']

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0 \
                    and accelerator.is_main_process and args.checkpoints_total_limit is not None:

                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            if os.path.exists(removing_checkpoint):
                                shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()
                    unwrap_net = accelerator.unwrap_model(model)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        unwrap_net.save_pretrained(save_path, max_shard_size="20GB")
                    logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("training completed!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_obs_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.load(args.config)

    # merge args
    args_dict = vars(args).copy()
    args_dict.pop("config", None)
    config_keys = set(config.keys())
    cli_keys = set(args_dict.keys())

    # check conflict
    conflict_keys = cli_keys & config_keys
    if conflict_keys:
        print(f"Args conflict: {conflict_keys}")

    # merge
    merged_config = OmegaConf.merge(OmegaConf.create(args_dict), config)
    args = merged_config

    # training
    main(args)
