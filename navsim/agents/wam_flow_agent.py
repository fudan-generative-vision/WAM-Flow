import os
import re
from typing import Any, List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from torchvision import transforms

# import navsim.common.file_ops as fops
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from flow_matching.data.navsim import resize_pad
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
from fudoki.eval_loop import CFGScaledModel
from fudoki.janus.models import VLChatProcessor, MultiModalityCausalLM
from fudoki.janus.models.heading_mlp import TrajectoryHeadingMLP

IMG_LEN = 576


def fmt_signed_2dec(val: float) -> str:
    val = round(val, 2)
    if val == 0:
        val = 0.0
    return f"{val:.2f}"


def map_command_to_direction(command: np.ndarray) -> str:
    idx = np.argmax(command)
    if idx == 0:
        return "left"
    elif idx == 1:
        return "straight"
    elif idx == 2:
        return "right"
    return "unknown"


def get_dtype(dtype):
    if "fp16" in dtype:
        return torch.float16
    elif "bf16" in dtype:
        return torch.bfloat16
    else:
        return torch.float32


def resize_pad(image, image_size=384):
    w, h = image.size
    if w <= 0 or h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    
    resize_scale = image_size / max(w, h)
    new_w = max(1, int(w * resize_scale))
    new_h = max(1, int(h * resize_scale))
    
    padding_color = (127, 127, 127)
    new_image = Image.new('RGB', (image_size, image_size), padding_color)
    
    if new_w <= 0 or new_h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    paste_x = (image_size - new_w) // 2
    paste_y = (image_size - new_h) // 2
    
    new_image.paste(image, (paste_x, paste_y))
    return new_image


def extract_num_list(text):
    """
    Robustly extract trajectory number list from string (fixed return 16 numbers)
    Completion rule: copy last valid (x,y) pair instead of filling 0 simply
    
    Args:
        text (str): Raw string with possible garbled characters/missing keywords
    
    Returns:
        list[float]: Extracted trajectory list with exactly 16 numbers
    """
    trajectory_numbers = []
    
    # Priority 1: Extract via keyword matching (compatible with original format)
    if "Trajectory:" in text and "Driving decision:" in text:
        trajectory_part = text.split("Trajectory:")[1]
        numbers_str = trajectory_part.split("Driving decision:")[0].strip()
        # Validate number sequence format
        if re.match(r'^-?\d+(\.\d+)?(,-?\d+(\.\d+)?)*$', numbers_str.replace(" ", "")):
            trajectory_numbers = [float(num.strip()) for num in numbers_str.split(",")]
    
    # Priority 2: Full text number extraction if keyword method fails
    if not trajectory_numbers:
        pred_str = text.strip()
        nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', pred_str)
        trajectory_numbers = [float(n) for n in nums]
    
    # Core logic: Complete numbers with last valid (x,y) pair (not 0)
    if len(trajectory_numbers) < 16:
        last_valid_x = None
        last_valid_y = None
        
        # Even length: last element is y, take last two as (x,y)
        if len(trajectory_numbers) % 2 == 0 and len(trajectory_numbers) >= 2:
            last_valid_x = trajectory_numbers[-2]
            last_valid_y = trajectory_numbers[-1]
        # Odd length: last element is x, reuse previous y or x as y
        elif len(trajectory_numbers) % 2 == 1:
            last_valid_x = trajectory_numbers[-1]
            last_valid_y = trajectory_numbers[-2] if len(trajectory_numbers) >= 2 else last_valid_x
        
        # Calculate numbers to complete
        need_complete = 16 - len(trajectory_numbers)
        
        # Complete with (x,y) pairs
        if need_complete > 0 and last_valid_x is not None and last_valid_y is not None:
            complete_nums = []
            for i in range(need_complete // 2):
                complete_nums.append(last_valid_x)
                complete_nums.append(last_valid_y)
            # Handle odd completion count (theoretical edge case)
            if need_complete % 2 == 1:
                complete_nums.append(last_valid_x)
            
            trajectory_numbers += complete_nums[:need_complete]
    
    # Truncate if over 16 numbers
    if len(trajectory_numbers) > 16:
        trajectory_numbers = trajectory_numbers[:16]
    # Fallback: return 16 zeros if no valid numbers
    elif len(trajectory_numbers) == 0:
        trajectory_numbers = [0.0] * 16
    
    return trajectory_numbers


class WamFlowAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5),
        fudoki_path: str = "",
        wam_flow_path: str = "",
        text_embedding_path: str = "",
        image_embedding_path: str = "",
        heading_mlp_path: str = "",
        discrete_fm_steps: int = 50,
        seed: int = 99,
        dtype: str = "default",
    ):
        super().__init__(trajectory_sampling)

        self.fudoki_path            = fudoki_path
        self.wam_flow_path          = wam_flow_path
        self.image_embedding_path   = image_embedding_path
        self.text_embedding_path    = text_embedding_path
        self.heading_mlp_path       = heading_mlp_path

        self.discrete_fm_steps      = discrete_fm_steps

        self.batch_size             = 1
        self.txt_max_length         = 500
        self.quantize_max_num       = 100 
        self.quantize_min_num       = -100 
        self.quantize_interval      = 0.01

        self.seed                   = seed
        self.dtype                  = get_dtype(dtype)

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = f"cuda:{local_rank}"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cudnn.benchmark = True

        self.transform_img = transforms.Compose([
            transforms.Lambda(resize_pad),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.fudoki_path)

        interval = int((self.quantize_max_num - self.quantize_min_num) / self.quantize_interval) + 1
        num_tokens = [f"{x:.2f}" for x in np.linspace(self.quantize_min_num, self.quantize_max_num, interval)]
        self.vl_chat_processor.tokenizer.add_tokens(num_tokens)

        model = MultiModalityCausalLM.from_pretrained(
            self.wam_flow_path
        ).to(self.device, dtype=self.dtype)
        model.eval()
        model.train(False)

        cfg_model = CFGScaledModel(model, g_or_u='understanding')

        path_txt = MixtureDiscreteSoftmaxProbPath(mode='text', embedding_path=self.text_embedding_path)
        path_img = MixtureDiscreteSoftmaxProbPath(mode='image', embedding_path=self.image_embedding_path)

        self.vocabulary_size_txt = max(len(self.vl_chat_processor.tokenizer), model.language_model.get_input_embeddings().weight.shape[0])
        with torch.no_grad():
            path_txt.set_embedding(model.language_model.get_input_embeddings())

            self.solver = MixtureDiscreteSoftmaxEulerSolver(
                model=cfg_model,
                path_txt=path_txt,
                path_img=path_img,
                vocabulary_size_txt=self.vocabulary_size_txt,
                vocabulary_size_img=model.config.gen_vision_config.params.image_token_size,
            )

        heading_model = TrajectoryHeadingMLP(hidden_dims=[512, 512, 256, 128])
        state_dict = torch.load(self.heading_mlp_path, map_location=self.device)
        heading_model.load_state_dict(state_dict)
        heading_model.eval()
        self.heading_model = heading_model.to(self.device)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[],
            lidar_pc=False,
        )

    def get_current_navigation_infomation(self, scene) -> Tuple[str, np.ndarray]:
        current_frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        navigation_command = current_frame.ego_status.driving_command
        
        navigation_command = np.array(navigation_command)

        direction = map_command_to_direction(navigation_command)
        return direction

    def get_ego_status(self, scene):
        start_frame_idx = scene.scene_metadata.num_history_frames - 1
        s = scene.frames[start_frame_idx].ego_status
        vel = s.ego_velocity.tolist()
        acc = s.ego_acceleration.tolist()
        
        vel_str = f"({fmt_signed_2dec(vel[0])},{fmt_signed_2dec(vel[1])})"
        acc_str = f"({fmt_signed_2dec(acc[0])},{fmt_signed_2dec(acc[1])})"

        return vel_str, acc_str

    def compute_trajectory(self, agent_input: AgentInput, scene=None) -> Trajectory:

        navigation_info = self.get_current_navigation_infomation(scene)
        state = self.get_ego_status(scene)

        img_path = str(agent_input.cameras[-1].cam_f0.image)
        img = Image.open(img_path).convert("RGB")

        init_pos_str = "(0.00,0.00)"

        with torch.no_grad():
            conversation = [
                {
                    "role": "User",
                    "content": "Here is front views of a driving vehicle:\n"
                    f"<image_placeholder>\n"
                    f"The navigation information is: {navigation_info}\n"
                    f"The current position is {init_pos_str}\n"
                    f"Current velocity is: {state[0]}  and current accelerate is: {state[1]}\n"
                    "Predict the optimal driving action for the next 4 seconds with 8 new waypoints."
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt=self.vl_chat_processor.system_prompt,
            )
            if '<image_placeholder>' in sft_format:
                img = self.transform_img(img)
                img_len = IMG_LEN
            else:
                img = None
                img_len = IMG_LEN

            input_ids = self.vl_chat_processor.tokenizer.encode(sft_format)
            input_ids = torch.LongTensor(input_ids)

            # add image tokens to the input_ids
            image_token_mask = (input_ids == self.vl_chat_processor.image_id)
            image_indices = image_token_mask.nonzero()
            input_ids, _ = self.vl_chat_processor.add_image_token(
                image_indices=image_indices,
                input_ids=input_ids,
            )

            # pad tokens
            rows_to_pad = max(self.txt_max_length + img_len - input_ids.shape[0], 100)
            input_ids = torch.cat([input_ids, torch.LongTensor([self.vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0)
            attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
            attention_mask[:] = True
            
            # obtain image token mask and fill in img token_ids
            if img is not None:
                image_expanded_token_mask = (input_ids == self.vl_chat_processor.image_id).to(dtype=int)
                image_expanded_mask_indices = torch.where(image_expanded_token_mask == 1)[0]
                input_ids[image_expanded_mask_indices] = 0
            else:
                image_expanded_token_mask = torch.zeros_like(input_ids)
            
            # obtain text token mask
            # We assume that there is only one turn for assistant to respond
            text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
            split_token = self.vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
            split_token_length = len(split_token)
            
            start_index = -1
            for j in range(len(input_ids) - split_token_length + 1):
                if input_ids[j:j + split_token_length].numpy().tolist() == split_token:
                    start_index = j
                    break

            if start_index != -1:
                text_expanded_token_mask[(start_index+split_token_length):] = 1
            else:
                raise ValueError("Split token not found in input_ids")

            generation_or_understanding_mask = 0
            data_info = dict()
            data_info['text_token_mask'] = text_expanded_token_mask.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
            data_info['image_token_mask'] = image_expanded_token_mask.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
            data_info['generation_or_understanding_mask'] = torch.Tensor([generation_or_understanding_mask]).unsqueeze(0).repeat(self.batch_size, 1).to(self.device).to(dtype=int)

            data_info['attention_mask'] = attention_mask.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
            data_info['sft_format'] = sft_format
            if img is not None:
                data_info['understanding_img'] = img.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).to(self.device)
                data_info['has_understanding_img'] = torch.Tensor([True]).to(dtype=int).unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
            else:
                data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(self.batch_size, 1, 1, 1).to(self.device)
                data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).repeat(self.batch_size, 1).to(self.device)

            x_0_txt = torch.randint(self.vocabulary_size_txt, input_ids.shape, dtype=torch.long, device=self.device)
            x_init = x_0_txt * data_info['text_token_mask'] + input_ids * (1 - data_info['text_token_mask'])

            synthetic_samples = self.solver.sample(
                x_init=x_init,
                step_size=1.0/self.discrete_fm_steps,
                return_intermediates=False,
                div_free=0,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=0,
            )
            
            sentences = self.vl_chat_processor.tokenizer.batch_decode(synthetic_samples, skip_special_tokens=True)[0]

        # print(f"======= sentences ===============\nsentences:{sentences}")

        nums = extract_num_list(sentences.strip())
        traj_xy = torch.tensor(nums, dtype=torch.float32).reshape(1, 8, 2).to(self.device)
        with torch.no_grad():
            heading_pred = self.heading_model(traj_xy).cpu().squeeze().numpy()

        traj_xy_np = traj_xy[0].cpu().numpy()
        poses = np.concatenate([traj_xy_np, heading_pred.reshape(-1, 1)], axis=1).astype(np.float32)

        # print(f"======= poses ====================\nposes:{poses}")
        return Trajectory(poses)