import os
import json
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset as HFDataset, load_dataset
from transformers import ImageProcessingMixin, PreTrainedTokenizer

from .template import ChatTemplate, TEMPLATE_REGISTRY

from specforge.data.preprocessing import OfflineEagle3Dataset

try:
    import yt.wrapper as yt
    import yt.yson as yson
    YT_AVAILABLE = True
except ImportError:
    print("Could not import YT requirements, will not be able to data points from YT")
    YT_AVAILABLE = False


def build_loss_mask(
    input_ids: torch.Tensor,
    start_seq: List[int],
    end_seq: Optional[List[int]]
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
    def get_triggers(input_ids: torch.Tensor, seq: Optional[List[int]]):
        if seq is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)

        triggers = torch.ones(len(input_ids) - len(seq), dtype=torch.bool)
        for idx, tok in enumerate(seq):
            end_idx = -len(seq)+idx
            triggers &= (input_ids[idx:end_idx] == tok)

        return torch.nonzero(
            torch.nn.functional.pad(triggers, pad=(len(seq), 0), value=False),
            as_tuple=True
        )[0]

    # 1. Detect Start and End Trigger Sequences
    start_triggers = get_triggers(input_ids, start_seq)
    end_triggers = get_triggers(input_ids, end_seq)

    # 2. Match End Trigger Sequencez Whenever Possible
    if any(end_triggers):
        end_insertion_indices = torch.searchsorted(end_triggers, start_triggers)
        valid_mask = end_insertion_indices < len(end_triggers)
        matched_end_triggers = end_triggers[end_insertion_indices[valid_mask]]
    else:
        matched_end_triggers = torch.zeros_like(input_ids, dtype=torch.bool)

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

    tools = kwargs.pop("tools", ["[]"]*len(conversations))
    reqids = kwargs.pop("reqids", [""]*len(conversations))
    kwargs_list = [{} for _ in range(len(conversations))]
    for key, value_list in kwargs.items():
        for i, value in enumerate(value_list):
            kwargs_list[i][key] = value
    for conv_str, tools_str, reqid, _ in zip(conversations, tools, reqids, kwargs_list):
        # This flag is required for OOM testing:
        # it builds dataset with input_ids with maximum length
        if os.environ.get("TESTING", "0") == "1":
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
            ).input_ids[0]
        else:
            try:
                input_ids = tokenizer.apply_chat_template(
                    conversation=json.loads(conv_str),
                    tools=json.loads(tools_str),
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                    return_tensors="pt"
                )[0]
            except Exception as e:
                # For easier debugging, dataset building can be launched with a single worker
                print(
                    f"Failed to preprocess row {reqid}"
                    f"\nconv_str: {conv_str}"
                    f"\ntools_str: {tools_str}"
                    f"\nreqid: {reqid}"
                )
                raise e

        start_seq = tokenizer(chat_template_inst.assistant_header).input_ids
        end_seq = (
            tokenizer(chat_template_inst.end_of_turn_token).input_ids
            if chat_template_inst.end_of_turn_token else None
        )
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
        extra_kwargs = {}
        if "tools" in examples and not is_preformatted:
            extra_kwargs["tools"] = examples["tools"]
        if "request_id" in examples:
            extra_kwargs["reqids"] = examples["request_id"]

        if is_preformatted:
            # Handle pre-formatted text (should be in "text" column)
            if "text" not in examples:
                raise ValueError(
                    f"Expected 'text' column for is_preformatted=True, but found columns: {list(examples.keys())}"
                )
            processed = preprocess_conversations(
                tokenizer=tokenizer,
                conversations=examples["text"],
                chat_template=chat_template,
                max_length=max_length,
                is_preformatted=True,
                **extra_kwargs
            )
        else:
            # Handle ShareGPT conversations
            if "messages" not in examples:
                raise ValueError(
                    f"Expected 'messages' column for is_preformatted=False, but found columns: {list(examples.keys())}"
                )
            processed = preprocess_conversations(
                tokenizer=tokenizer,
                conversations=examples["messages"],
                chat_template=chat_template,
                max_length=max_length,
                is_preformatted=False,
                **extra_kwargs
            )

        return processed

    def filter_preprocessed(results):
        return (
            torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x).squeeze() for x in results["loss_mask"]],
                batch_first=True,
                padding_side="left",
                padding_value=0
            ).sum(dim=-1) > 0
        ).tolist()

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

    batch_size = 1000

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


class OfflineEagle3YTTableDataset(OfflineEagle3Dataset):
    def __init__(self, datapath, transform=None, max_len=2048, seed=None):
        assert datapath.startswith("yt:"), "YT source should start with 'yt:'"

        super().__init__([datapath], transform, max_len)
        self.seed = seed

        self.yt_config = yt.default_config.get_config_from_env()
        # self.yt_config["enable_rpc_proxy_in_job_proxy"] = True
        # self.yt_config["backend"] = "rpc"
        self.yt_config["read_parallel"]["enable"] = True
        self.yt_config["read_parallel"]["max_thread_count"] = 32

        self.proxy, self.yt_table_path = datapath[3:].split("/", 1)
        self._reset_reader(init=True)


    def _reset_reader(self, init=False):
        if not init:
            del self.reader
            del self.yt_client

        # table_reader = {"window_size": 5368709120, "max_buffer_size": 10737418240}  # 5 GiB / 10 GiB
        # if self.seed:
        #     table_reader.update({"sampling_mode": "row", "sampling_rate": 1, "sampling_seed": self.dp_seed})

        self.yt_client = yt.YtClient(proxy=self.proxy, token=os.environ['YT_TOKEN'], config=self.yt_config)
        self.reader = self.yt_client.read_table(self.yt_table_path, enable_read_parallel=True)
        self.index = 0

        if init:
            self.total_rows = self.yt_client.get_attribute(self.yt_table_path, "path_count")
            print(f"Found {self.total_rows} in training YT table")

    @property
    def dp_seed(self):
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        return self.seed + world_size * self._epoch + rank

    def __len__(self):
        return self.total_rows

    def _open_file(self):
        buffer = BytesIO()
        try:
            while True:
                row = next(self.reader)

                # print(f"RANK {torch.distributed.get_rank()}: INDEX {self.index} -> {row["file_path"]}:{row["index"]}")
                self.index += 1

                buffer.write((yson.get_bytes(row["data"])))
                if not row["has_next"]:
                    # print(f"RANK {torch.distributed.get_rank()}: HIT END")
                    break

            buffer.seek(0)
            return torch.load(buffer, weights_only=False)
        except Exception as e:
            print(f"ERROR Failed to load {row["file_path"]} with error {e}")
            raise e

    def __getitem__(self, index):
        return self.process_data(self._open_file(), self.max_len, self.transform)

    def set_epoch(self, epoch):
        self._epoch = epoch
        self._reset_reader()


def build_offline_eagle3_yt_table_dataset(
    hidden_states_path: str,
    max_len: int = 2048,
    seed: Optional[int] = None
) -> torch.utils.data.Dataset:
    if not YT_AVAILABLE:
        raise ImportError("Error while importing YT requirements, try running `pip imstall -r requirements_yt.txt`")

    return OfflineEagle3YTTableDataset(
        hidden_states_path,
        max_len=max_len,
        seed=seed
    )


# Unline list_local_files, does not expect any subdirectories
def list_yt_files(client: yt.YtClient, path: str, suffixes=[".ckpt"]):
    datapaths = [os.path.join(path, file) for file in client.list(path)]
    for suffix in suffixes:
        datapaths = [f_name for f_name in datapaths if f_name.endswith(suffix)]
    return datapaths


class OfflineEagle3YTDataset(OfflineEagle3Dataset):
    def __init__(self, datapath, transform=None, max_len=2048):
        super().__init__(datapath, transform, max_len)

        self.yt_client = yt.YtClient(proxy="hahn", token=os.environ['YT_TOKEN'])

    def _open_file(self, index):
        # print(f"RANK {torch.distributed.get_rank()}: Trying to load {self.datapaths[index]} ({index})")
        data = self.yt_client.read_file(path=self.datapaths[index], enable_read_parallel=True)
        return torch.load(BytesIO(initial_bytes=data.read()), weights_only=False)


def build_offline_eagle3_yt_dataset(
    hidden_states_path: str,
    max_len: int = 2048
) -> torch.utils.data.Dataset:
    if not YT_AVAILABLE:
        raise ImportError("Error while importing YT requirements, try running `pip imstall -r requirements_yt.txt`")

    assert hidden_states_path.startswith("yt:"), "YT source should start with 'yt:'"
    proxy, yt_table_path = hidden_states_path[3:].split("/", 1)
    yt_client = yt.YtClient(proxy=proxy, token=os.environ['YT_TOKEN'])

    return OfflineEagle3YTDataset(
        list_yt_files(yt_client, yt_table_path),
        max_len=max_len
    )


def multi_load_dataset(dataset_path: str, columns: List[str]) -> HFDataset:
    if dataset_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=dataset_path)["train"]
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in columns]
        )
    elif dataset_path.endswith(".tsv"):
        dataset = load_dataset("csv", data_files=dataset_path, delimiter="\t")["train"]
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in columns]
        )
    elif dataset_path.endswith(".json"):
        dataset = load_dataset("json", data_files=dataset_path)["train"]
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in columns]
        )
    else:
        raise ValueError(f"Could not find loading method for path {dataset_path}")

    return dataset
