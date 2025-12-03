import os
import json
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset as HFDataset
from transformers import ImageProcessingMixin, PreTrainedTokenizer

from .format_openai import (
    format_openai_messages_to_yagpt,
    format_openai_tools_to_yagpt
)
from .template import ChatTemplate, TEMPLATE_REGISTRY


# This flag is required for OOM testing:
# it builds dataset with input_ids with maximum length
TESTING = False


def build_loss_mask(
    input_ids: torch.Tensor,
    start_seq: List[int],
    end_seq: List[int]
) -> torch.Tensor:
    """
    Builds a boolean mask where True marks tokens inside a conversation response.
    
    The mask becomes True AFTER the start sequence.
    The mask remains True UNTIL AND INCLUDING the end sequence.
    
    Args:
        input_ids: Tensor of shape (batch_size, seq_len) or (seq_len).
        
    Returns:
        torch.BoolTensor of the same shape as input_ids.
    """
    # Ensure input is at least 2D for consistent processing
    assert input_ids.ndim == 1

    if len(input_ids) < len(start_seq):
        # Sequence too short to contain start pattern
        return torch.zeros_like(input_ids, dtype=torch.long)

    # Detect Sequence: The mask should turn ON at the token *after* the token sequence.
    # We pad len(seq) positions to the left (and 0 to right) to shift the trigger forward.
    def get_triggers(input_ids: torch.Tensor, seq: List[int]):
        triggers = torch.ones(len(input_ids) - len(seq), dtype=torch.bool)
        for idx, tok in enumerate(seq):
            end_idx = -len(seq)+idx
            triggers &= (input_ids[idx:end_idx] == tok)

        return torch.nonzero(
            torch.nn.functional.pad(
                triggers,
                pad=(3, 0),
                value=False
            ),
            as_tuple=True
        )[0]

    # 1. Detect Start and End Trigger Sequences
    start_triggers = get_triggers(input_ids, start_seq)
    end_triggers = get_triggers(input_ids, end_seq)

    # 2. Match End Trigger Sequencez Whenever Possible
    end_insertion_indices = torch.searchsorted(end_triggers, start_triggers)
    valid_mask = end_insertion_indices < len(end_triggers)
    matched_end_triggers = end_triggers[end_insertion_indices[valid_mask]]

    # 3. Update the Delta Tensor
    # Add +1 at start positions and subtract -1 at matched end positions
    mask_delta = torch.zeros_like(input_ids)
    mask_delta[start_triggers] += 1
    mask_delta[matched_end_triggers] -= 1

    # Return the loss mask
    return (torch.cumsum(mask_delta, dim=0) > 0).long()


# Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py
def preprocess_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: Union[List[Dict[str, Any]], List[str]],
    chat_template: str,
    max_length: int = 2048,
    is_preformatted: bool = False,
    **kwargs,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess a batch of ShareGPT style conversations or pre-formatted text.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        conversations: A list of conversations (if is_preformatted=False) or
                      a list of pre-formatted text strings (if is_preformatted=True).
        chat_template: The chat template to use for formatting/identifying spans.
        max_length: The maximum length of the tokenized input.
        is_preformatted: Whether the input is already formatted text strings.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - loss_mask: List of loss masks indicating which tokens should contribute to the loss.
            - attention_mask: List of attention masks.
    """
    chat_template_inst: ChatTemplate = TEMPLATE_REGISTRY.get(chat_template.split("_")[0])

    # prepare result
    results = {"input_ids": [], "loss_mask": [], "attention_mask": []}

    tools = kwargs.pop("tools", [[]]*len(conversations))
    kwargs_list = [{} for _ in range(len(conversations))]
    for key, value_list in kwargs.items():
        for i, value in enumerate(value_list):
            kwargs_list[i][key] = value
    for conv_str, tools_str, _ in zip(conversations, tools, kwargs_list):
        if TESTING:
            results["input_ids"].append(
                torch.randint(0, len(tokenizer), (1, max_length), dtype=torch.long)
            )
            results["loss_mask"].append(torch.ones((1, max_length), dtype=torch.long))
            results["attention_mask"].append(torch.ones((1, max_length), dtype=torch.long))
            continue

        if is_preformatted:
            input_ids = tokenizer(
                conv_str,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
        else:
            conv_dict = json.loads(conv_str)
            tools_dict = json.loads(tools_str)
            if "openai" in chat_template:
                conv_dict = format_openai_messages_to_yagpt(conv_dict)
                tools_dict = format_openai_tools_to_yagpt(tools_dict)

            input_ids = tokenizer.apply_chat_template(
                conversation=conv_dict,
                tools=tools_dict,
                add_generation_prompt=False,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids

        start_seq = tokenizer(chat_template_inst.assistant_header).input_ids
        end_seq = tokenizer(chat_template_inst.end_of_turn_token).input_ids
        loss_mask = build_loss_mask(input_ids, start_seq, end_seq)

        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: Optional[str] = None,
    max_length: Optional[int] = 2048,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 8,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    is_vlm: Optional[bool] = False,
    processor: Optional[ImageProcessingMixin] = None,
    is_preformatted: Optional[bool] = False,
) -> HFDataset:
    """
    build eagle3 dataset

    Args:
        dataset: HF dataset to process.
        tokenizer: The tokenizer to use for tokenization.
        chat_template: The chat template to use for formatting conversations.
                        This includes the system prompt and user/assistant tokens
                        required to delineate different parts of the conversation
                        for loss mask generation.
        max_length: The maximum length of the tokenized input.
        shuffle_seed: The seed for shuffling the dataset.
        num_proc: The number of processes to use for multiprocessing.
        cache_dir: The directory to use for caching the processed dataset.
        cache_key: The key to use for caching the processed dataset.
        is_vlm: Whether the dataset is for VLM models.
        processor: The image processor to use for processing images.
        is_preformatted: Whether the dataset contains preformatted text of the conversation
                        (e.g. includes system prompt, user and assistant start and end tokens)
                        and doesn't need to have the chat template applied.
                        Note that the chat_template still needs to be specified to determine
                        the assistant spans for loss mask generation.
                        If True, expects "text" column with ready-to-train text.
                        If False, expects "conversations" column with ShareGPT format.

    Returns:
        The processed HF dataset.
    """
    if is_vlm:
        assert False, "vlm mode is not supported for custom preprocessing"

    dataset = dataset.shuffle(seed=shuffle_seed)
    original_cols = dataset.column_names

    def preprocess_function(examples):
        if is_preformatted:
            # Handle pre-formatted text (should be in "text" column)
            if "text" not in examples:
                raise ValueError(
                    f"Expected 'text' column for is_preformatted=True, but found columns: {list(examples.keys())}"
                )
            processed = preprocess_conversations(
                tokenizer=tokenizer,
                messages=examples["text"],
                tools=[list()]*len(examples["text"]),
                chat_template=chat_template,
                max_length=max_length,
                is_preformatted=True,
            )
        else:
            # Handle ShareGPT conversations
            if "messages" not in examples:
                raise ValueError(
                    f"Expected 'messages' column for is_preformatted=False, but found columns: {list(examples.keys())}"
                )
            processed = preprocess_conversations(
                tokenizer=tokenizer,
                messages=examples["messages"],
                tools=examples["tools"],
                chat_template=chat_template,
                max_length=max_length,
                is_preformatted=False,
            )

        return processed

    def filter_preprocessed(results):
        print(results["loss_mask"].shape)
        raise Exception()
        return [(sum(mask[0]) > 0) for mask in results["loss_mask"]]

    # Process dataset only once
    if cache_dir and cache_key:
        load_from_cache_file = True
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.join(cache_dir, f"{cache_key}.pkl")
        print(f"dataset is cached at {cache_file_name}")
    elif cache_dir is None and cache_key is None:
        load_from_cache_file = False
        cache_file_name = None
        print(f"dataset is not cached")
    else:
        warnings.warn(
            f"cache_dir and cache_key must be provided together to make caching work"
        )

    # adjust batch size based on dataset type
    if is_vlm:
        batch_size = (
            200  # reduce batch size for VLM datasets to avoid PyArrow offset overflow
        )
    else:
        batch_size = 1000  # default for conversations

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=original_cols,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
    )
    dataset = dataset.filter(
        filter_preprocessed,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size
    )

    dataset.set_format(type="torch")
    return dataset


# ==============================
# Vocab Mapping
# ==============================
def generate_vocab_mapping_file(
    dataset: HFDataset,
    target_vocab_size: int,
    draft_vocab_size: int,
    cache_dir: str = "./cache/vocab_mapping",
    cache_key: str = "vocab_mapping",
    num_proc: int = 1
) -> str:
    """
    Generate a vocab mapping file for the dataset.

    Args:
        dataset: The dataset to process.
        target_vocab_size: The target vocabulary size.
        draft_vocab_size: The draft vocabulary size.
        cache_dir: The directory to use for caching the vocab mapping file.
        cache_key: The key to use for caching the vocab mapping file.

    Returns:
        The path to the vocab mapping file.
    """
    # prepare cache direcotory
    os.makedirs(cache_dir, exist_ok=True)
    vocab_mapping_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(vocab_mapping_path):
        print(f"Loading vocab mapping from the cached file at: {vocab_mapping_path}")
        return vocab_mapping_path

    original_cols = dataset.column_names
    batch_size = 1000
    processed_dataset = dataset.map(
        count_tokens,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=original_cols
    )

    token_dict = Counter()
    for partial_dict in processed_dataset["token_dict"]:
        token_dict.update({int(tok): freq for tok, freq in json.loads(partial_dict).items()})

    # generate the d2t and t2d mapping
    d2t, t2d = process_token_dict_to_mappings(
        token_dict,
        draft_vocab_size,
        target_vocab_size,
    )

    vocab_mapping = {
        "d2t": d2t,
        "t2d": t2d,
    }
    torch.save(vocab_mapping, vocab_mapping_path)
    print(f"Saved vocab mapping to: {vocab_mapping_path}")
    return vocab_mapping_path


def count_tokens(items: HFDataset) -> Counter:
    input_ids = items["input_ids"]
    loss_mask = items["loss_mask"]
    masked_ids = input_ids[loss_mask == 1]
    unique_ids, counts = masked_ids.unique(return_counts=True)
    batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))
    return {"token_dict": [json.dumps(batch_token_dict)]}


def process_token_dict_to_mappings(
    token_dict: Counter,
    draft_vocab_size: int,
    target_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process token_dict to create d2t and t2d mappings, with optional caching.

    Args:
        token_dict: A Counter object mapping token ids to their frequencies.
        draft_vocab_size: The size of the draft vocabulary.
        target_vocab_size: The size of the target vocabulary.

    Returns:
        A tuple containing:
            - d2t: A tensor mapping draft token ids to target token ids.
            - t2d: A tensor mapping target token ids to draft token ids.
    """
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        missing_tokens = set(range(draft_vocab_size)) - existing_tokens
        for token in missing_tokens:
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break
    print(f"Added missing tokens to reach draft vocab size: {draft_vocab_size}")
    print(f"Total tokens after addition: {len(token_dict)}")
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)

    if total_frequency == 0:
        print(
            "Warning: Total token frequency is zero. All tokens will have zero ratio."
        )
        top_N_ratio = 0.0
    else:
        top_N_ratio = top_N_frequency_sum / total_frequency

    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()

    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(target_vocab_size)]
    d2t = torch.tensor(d2t)
    t2d = torch.tensor(t2d)

    return d2t, t2d
