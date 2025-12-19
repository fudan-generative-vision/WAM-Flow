# LINT_ME
import os
import argparse

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torchvision import transforms
from torch.backends import cudnn
from transformers import set_seed

from flow_matching.data.navsim import resize_pad
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
from fudoki.eval_loop import CFGScaledModel
from fudoki.janus.models import VLChatProcessor
from fudoki.model import instantiate_model


VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the script with custom arguments.")
    parser.add_argument(
        "--seed", type=int, default=999, 
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for processing."
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, 
        help="Path to the checkpoint directory."
    )
    parser.add_argument(
        "--processor_path", type=str, required=True, 
        help="Path to the processor."
    )
    parser.add_argument(
        "--text_embedding_path", type=str, required=True, 
        help="Path to the text embedding."
    )
    parser.add_argument(
        "--image_embedding_path", type=str, required=True, 
        help="Path to the image embedding."
    )
    parser.add_argument(
        "--discrete_fm_steps", type=int, default=5, 
        help="Inference steps for discrete flow matching"
    )
    parser.add_argument(
        "--txt_max_length", type=int, default=500, 
        help="Text length maximum"
    )
    parser.add_argument(
        "--image_paths", type=str, required=True, 
        help="Path to the input image."
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, 
        help="Directory to save the output files."
    )
    return parser.parse_args()


def extract_number_pairs(input_str, k=None):
    """
    Extract number pairs from a string and form tuples of (float(x), float(y))
    
    Parameters:
        input_str (str): Input string in the format "4.78,-0.01,9.69,-0.01,..."
        k (int, optional): Number of tuples to extract. If None, extract all possible tuples
    
    Returns:
        tuple: Contains two elements
            - list: List of successfully extracted tuples
            - list: List of error messages
    """
    pairs = []
    errors = []

    # Check if input is empty or not a string
    if not input_str or not isinstance(input_str, str):
        errors.append("Input is not a valid string")
        return pairs, errors

    # Split the string
    elements = input_str.split(',')

    # Process each element and attempt to convert to float
    numbers = []
    for index, elem in enumerate(elements):
        # Remove possible whitespace characters
        elem_clean = elem.strip()
        if not elem_clean:
            errors.append(f"Empty value at position {index}")
            continue

        try:
            num = float(elem_clean)
            numbers.append(num)
        except ValueError:
            errors.append(f"Value '{elem_clean}' at position {index} \
                            cannot be converted to a number")

    # Calculate maximum possible pairs
    max_possible = len(numbers) // 2

    # Determine number of pairs to extract
    if k is None:
        # Extract all possible pairs
        num_to_extract = max_possible
    else:
        # Ensure k is a positive integer
        try:
            k = int(k)
            if k <= 0:
                errors.append(f"k value {k} must be a positive integer")
                return pairs, errors
            num_to_extract = min(k, max_possible)
        except (ValueError, TypeError):
            errors.append(f"k value {k} is not a valid integer")
            return pairs, errors

    # Extract number pairs
    for i in range(num_to_extract):
        x = numbers[2*i]
        y = numbers[2*i + 1]
        pairs.append((x, y))

    # Check for unpaired numbers
    if len(numbers) % 2 != 0:
        errors.append(f"There are {len(numbers) % 2} unpaired number(s)")

    # Check if requested k value was achieved
    if k is not None and num_to_extract < k:
        errors.append(f"Only {num_to_extract} valid number pairs can be extracted, \
                        less than the requested {k}")

    return pairs, errors


def main():
    args = parse_arguments()

    dist.init_process_group(
        "nccl",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    set_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device(f"cuda:{local_rank}")

    image_paths = args.image_paths.split(',')

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.processor_path)
    num_tokens_length = 0
    num_tokens = [f"{x:.2f}" for x in np.linspace(-100, 100, 20001)]
    num_tokens_length = len(num_tokens)
    vl_chat_processor.tokenizer.add_tokens(num_tokens)

    model = instantiate_model(args.checkpoint_path).to(device, dtype=torch.float32)
    model.train(False)

    batch_size = args.batch_size
    discrete_fm_steps = args.discrete_fm_steps
    txt_max_length = args.txt_max_length

    cfg_weighted_model = CFGScaledModel(model=model, g_or_u='understanding')
    with torch.no_grad():
        path_txt = MixtureDiscreteSoftmaxProbPath(
            mode='text',
            embedding_path=args.text_embedding_path
        )
        path_txt.set_embedding(model.language_model.get_input_embeddings())
        path_img = MixtureDiscreteSoftmaxProbPath(
            mode='image',
            embedding_path=args.image_embedding_path
        )
        solver = MixtureDiscreteSoftmaxEulerSolver(
            model=cfg_weighted_model,
            path_txt=path_txt,
            path_img=path_img,
            vocabulary_size_txt=VOCABULARY_SIZE_TXT + num_tokens_length,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )

        imgs = []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            transform = transforms.Compose([
                transforms.Lambda(resize_pad),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            imgs.append(transform(img))

        if len(imgs) > 0:
            imgs = torch.stack(imgs, dim=0)   # [N, C, H, W]
            img_len = len(imgs) * IMG_LEN
        else:
            imgs = None
            img_len = IMG_LEN  # default

        generation_understanding_indicator = 0 # this is an understanding sample
        conversation = [
            {
                "role": "User",
                "content": (
                    "Here is front views of a driving vehicle:\n<image_placeholder>\n"
                    "The navigation information is: straight\n"
                    "The current position is (0.00,0.00)\n"
                    "Current velocity is: (8.34,0.18)  and current accelerate is: (-0.83,0.28)\n"
                    "Predict the optimal driving action for the next 4 seconds with 8 new waypoints."
                )
            },
            {
                "role": "Assistant",
                "content": ""
            } # "3.88,-0.06,7.50,-0.07,10.86,-0.10,13.95,-0.11,16.75,-0.13,19.29,-0.15,21.60,-0.12,23.67,-0.11"
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=vl_chat_processor.system_prompt,
        )

        # tokenize
        input_ids = vl_chat_processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)
        # add image tokens to the input_ids
        image_token_mask = input_ids == vl_chat_processor.image_id
        image_indices = image_token_mask.nonzero()
        assert len(image_indices) == len(image_paths), \
                f"Number of images ({len(image_paths)}) \
                  does not match the number of image tokens ({len(image_indices)})"

        input_ids, _ = vl_chat_processor.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # pad tokens
        original_input_id_len = input_ids.shape[0]
        if original_input_id_len >= txt_max_length + img_len:
            raise ValueError("Sentences too long, not supported so far...")

        rows_to_pad = txt_max_length + img_len - input_ids.shape[0]
        input_ids = torch.concat([
            input_ids,
            torch.LongTensor([vl_chat_processor.pad_id]).repeat(rows_to_pad)
        ], dim=0)
        attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
        attention_mask[:] = True

        # obtain image token mask and fill in img token_ids
        if imgs is not None:
            image_expanded_token_mask = (input_ids == vl_chat_processor.image_id).to(dtype=int)
            image_expanded_mask_indices = torch.where(image_expanded_token_mask == 1)[0]
            input_ids[image_expanded_mask_indices] = 0
        else:
            image_expanded_token_mask = torch.zeros_like(input_ids)

        # obtain text token mask
        # We assume that there is only one turn for assistant to respond
        text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
        split_token = vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
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

        generation_or_understanding_mask = generation_understanding_indicator
        data_info = {}
        data_info['text_token_mask'] = text_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['image_token_mask'] = image_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['generation_or_understanding_mask'] = \
            torch.Tensor([generation_or_understanding_mask]).unsqueeze(0).repeat(batch_size, 1).to(device).to(dtype=int)

        data_info['attention_mask'] = attention_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['sft_format'] = sft_format
        if imgs is not None:
            data_info['understanding_img'] = imgs.unsqueeze(0).to(device, dtype=torch.float32).repeat(batch_size, 1, 1, 1, 1)
            data_info['has_understanding_img'] = torch.Tensor([True]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)
        else:
            data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).repeat(batch_size, 1).to(device)


        x_0_txt = torch.randint(VOCABULARY_SIZE_TXT + num_tokens_length, input_ids.shape, dtype=torch.long, device=device)
        x_init = x_0_txt * data_info['text_token_mask'] + input_ids * (1 - data_info['text_token_mask'])

        synthetic_samples = solver.sample(
            x_init=x_init,
            step_size=1.0/discrete_fm_steps,
            verbose=True,
            return_intermediates=False,
            div_free=0,
            dtype_categorical=torch.float32,
            datainfo=data_info,
            cfg_scale=0,
        )
        sentence = vl_chat_processor.tokenizer.batch_decode(
            synthetic_samples,
            skip_special_tokens=True
        )[0]
        print("Sentence:", sentence)

        waypoint = extract_number_pairs(sentence, k=8)[0]
        print("Waypoint: ", waypoint)


if __name__ == "__main__":
    main()
