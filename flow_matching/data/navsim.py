# LINT_ME
import os
import json
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from fudoki.janus.models import VLChatProcessor


VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576


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


class SupervisedDataset(Dataset):
    """
    Dataset for supervised training on image-text conversation data.

    Loads image-text conversation samples from JSON/JSONL files, processes images (resize/pad/normalize),
    tokenizes text prompts with image placeholders, and formats data for training.
    """
    def __init__(
        self,
        data_list: list,
        vl_chat_processor: VLChatProcessor,
        txt_max_length=500
    ):
        super().__init__()
        self.vl_chat_processor = vl_chat_processor
        self.txt_max_length = txt_max_length
        self.list_data_dict = []

        self.split_token = None

        self.transform_img = transforms.Compose([
            transforms.Lambda(resize_pad),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        for path in data_list:
            jsonl_list = []
            ext = os.path.splitext(path)[-1].lower()

            with open(path, "r", encoding="utf-8") as f:
                if ext == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if line:
                            jsonl_list.append(json.loads(line))
                elif ext == ".json":
                    data = json.load(f)
                    if isinstance(data, list):
                        jsonl_list.extend(data)
                    else:
                        jsonl_list.append(data)
                else:
                    raise ValueError(f"Unsupported file extension: {ext}")

            self.list_data_dict.extend(jsonl_list)


    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            print(f"[Warning] Error loading index {i}: {e}")
            # random choice
            rand_idx = np.random.randint(0, len(self.list_data_dict))
            print(f"[Retry] Loading random index {rand_idx} instead.")
            return self.__getitem__(rand_idx)


    def _get_item(self, i):
        item = self.list_data_dict[i]

        if "image" not in item:
            raise ValueError("Currently only image-based samples are supported.")

        image_list = item["image"]
        conversation = item["conversations"]

        if conversation[0]["from"] == "system":
            addition_system_prompt = conversation[0]["value"]
            conversation = conversation[1:]
        else:
            addition_system_prompt = ""

        # conv_length = len(conversation)
        # assert conv_length == 2, "only support single turn"

        conversation = self.construct_conv(conversation)

        data_dict = self.process_image_item(
            image_list,
            conversation,
            system_prompt=addition_system_prompt,
            txt_max_length=self.txt_max_length
        )
        return data_dict

    def construct_conv(self, conv):
        _conv = []
        for item in conv:
            role = item["from"]
            content = item["value"]
            _item = {}

            if role == "human":
                _item["role"] = "User"
            elif role == "gpt":
                _item["role"] = "Assistant"
            else:
                raise ValueError("role must be human or gpt")

            if "<image>" in content:
                content = content.replace("<image>", "<image_placeholder>")

            _item["content"] = content

            _conv.append(_item)

        return _conv

    def _find_split_token(self, input_ids, split_token_length):
        # start index for "Assistant:"
        start_index = -1
        for j in range(len(input_ids) - split_token_length, 0, -1):
            if input_ids[j:j + split_token_length].numpy().tolist() == self.split_token:
                start_index = j
                break
        return start_index

    def process_image_item(
        self,
        image_paths,
        conversation,
        system_prompt="",
        txt_max_length=500
    ):
        imgs = []
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            imgs.append(self.transform_img(img))

        if len(imgs) > 0:
            imgs = torch.stack(imgs, dim=0)   # [N, C, H, W]
            img_len = len(imgs) * IMG_LEN
        else:
            imgs = None
            img_len = 0  # default

        generation_understanding_indicator = 0

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.vl_chat_processor.system_prompt + system_prompt,
        )

        # tokenize
        input_ids = self.vl_chat_processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)

        # add image tokens to the input_ids
        image_token_mask = input_ids == self.vl_chat_processor.image_id
        image_indices = image_token_mask.nonzero()
        assert len(image_indices) == len(image_paths), \
            f"Number of images ({len(image_paths)}) does not match number of image tokens ({len(image_indices)})"

        input_ids, _ = self.vl_chat_processor.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # pad tokens
        if input_ids.shape[0] >= txt_max_length + img_len:
            rows_to_pad = random.randint(0, 50)
        else:
            rows_to_pad = txt_max_length + img_len - input_ids.shape[0]
        input_ids = torch.cat([input_ids, torch.LongTensor([self.vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0)
        attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
        attention_mask[:] = True

        # obtain image token mask and fill in img token_ids
        if imgs is not None:
            image_expanded_token_mask = (input_ids == self.vl_chat_processor.image_id).to(dtype=int)
            image_expanded_mask_indices = torch.where(image_expanded_token_mask == 1)[0]
            input_ids[image_expanded_mask_indices] = 0
        else:
            image_expanded_token_mask = torch.zeros_like(input_ids)

        # obtain text token mask
        # support multi turn, indicating the last one
        if self.split_token is None:
            self.split_token = self.vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
        split_token_length = len(self.split_token)
        start_index = self._find_split_token(input_ids, split_token_length)

        text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
        if start_index != -1:
            text_expanded_token_mask[(start_index+split_token_length):] = 1
        else:
            raise ValueError("Split token not found in input_ids")


        generation_or_understanding_mask = generation_understanding_indicator
        data_info = {}
        data_info['text_token_mask'] = text_expanded_token_mask
        data_info['image_token_mask'] = image_expanded_token_mask
        data_info['generation_or_understanding_mask'] = torch.Tensor([generation_or_understanding_mask])

        data_info['attention_mask'] = attention_mask
        data_info['sft_format'] = sft_format

        data_info['understanding_img'] = imgs
        data_info['has_understanding_img'] = torch.Tensor([True]).to(dtype=int)

        data_info["input_ids"] = torch.LongTensor(input_ids)

        # print("\n\n\n", sft_format)
        # target = self.vl_chat_processor.tokenizer.batch_decode(input_ids[text_expanded_token_mask == 1])
        # print("\n \n", ''.join(target).strip())
        # exit()
        return data_info
