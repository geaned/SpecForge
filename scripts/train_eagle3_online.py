import hashlib
import json
import math
import os
import time
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import configargparse
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
)
from specforge.data import (
    generate_vocab_mapping_file,
    multi_load_dataset,
    prepare_dp_dataloaders,
    yt_pull_dataset
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_device_mesh,
    init_distributed,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)

try:
    import nirvana.job_context as nv
    print_with_rank("Nirvana job context imported successully!")
    NV_JOB_CONTEXT = True
except ImportError:
    NV_JOB_CONTEXT = False

NV_EXT_SUFFIX = "_extracted"


def save_model(model, state, output_dir):
    # Save the model
    if dist.get_rank() == 0:
        if NV_JOB_CONTEXT:
            tar_output_dir = output_dir.split("/")[-1]
            output_dir = tar_output_dir + NV_EXT_SUFFIX
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        model_state_dict = model.state_dict()
        draft_model_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in model_state_dict.items()
            if "draft_model." in k and "embed" not in k.lower()
        }

        if dist.get_rank() == 0:
            torch.save(
                state,
                os.path.join(output_dir, "training_state.pt"),
            )
            print_on_rank0(
                f"Saved full training state to {output_dir}/training_state.pt"
            )
            model.draft_model.save_pretrained(
                output_dir,
                state_dict=draft_model_state_dict,
            )
            print_on_rank0(f"Saved model configuration to {output_dir}")
        
            if NV_JOB_CONTEXT:
                print_on_rank0(f"Saving compressed output to {tar_output_dir}...")
                subprocess.run(['tar', '-cvf', tar_output_dir, '-C', output_dir, '.'])
        dist.barrier()


def parse_args():
    params_path = None

    if NV_JOB_CONTEXT:
        nv_ctx = nv.context()
        inputs = nv_ctx.get_inputs()
        outputs = nv_ctx.get_outputs()
        nv_params = nv_ctx.get_parameters()
        params_path = inputs.get('params')

    if not params_path:
        params_path = os.environ.get("YAML_CONFIG_PATH", "")

    config_files = []
    if params_path:
        print(f"Reading config from {params_path}...")
        config_files.append(params_path)

    parser = configargparse.ArgumentParser(
        default_config_files=config_files,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description="Train Eagle3 with online data"
    )
    nv_inputs_disabled = not NV_JOB_CONTEXT

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=nv_inputs_disabled)
    parser.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    parser.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )

    # add training-related arguments
    parser.add_argument("--yt-token", type=str, default=None)
    parser.add_argument("--train-data-path", type=str, required=nv_inputs_disabled)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1, help="deprecated: use --draft-micro-batch-size")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.015)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--log-steps", type=int, default=50, help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )

    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--draft-global-batch-size", type=int, default=8)
    parser.add_argument(
        "--draft-micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size for draft model",
    )
    parser.add_argument("--draft-accumulation-steps", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=nv_inputs_disabled)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument(
        "--save-strategy", type=str, default="epochs", choices=["epochs", "steps"]
    )
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument("--attention-backend", type=str, default="flex_attention")

    # resume
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "swanlab", "mlflow", "print", "none"],
        help="The integration to report results and logs to.",
    )
    # wandb-specific args
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
    # swanlab-specific args
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default=None,
        help="The project name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-name",
        type=str,
        default=None,
        help="The experiment name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-key",
        type=str,
        default=None,
        help="The API key for swanlab non-interactive login.",
    )
    # mlflow-specific args
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="The MLflow run name. If not set, MLflow will auto-generate one.",
    )

    # vlm related args
    parser.add_argument(
        "--min-pixels", type=int, default=50176
    )  # 64*28*28 for qwen2.5-vl
    parser.add_argument(
        "--max-pixels", type=int, default=802816
    )  # 1024*28*28 for qwen2.5-vl

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-memory-steps", type=int, default=8)

    args = parser.parse_args()

    if NV_JOB_CONTEXT:
        args.target_model_path = inputs.get('model')
        args.draft_model_config = inputs.get('drafter_config')
        args.yt_token = nv_params.get('yt-token')
        train_data = json.load(open(inputs.get('train_data')))
        args.train_data_path = f"yt:{train_data['cluster']}/{train_data['table']}"
        eval_data = json.load(open(inputs.get('eval_data')))
        args.eval_data_path = f"yt:{eval_data['cluster']}/{eval_data['table']}"
        args.output_dir = outputs.get('spec_model') + NV_EXT_SUFFIX
        # TODO: logs to separate output

    if not args.yt_token:
        args.yt_token = os.environ.get("YT_TOKEN")

    return parser, args


def main(args=None):
    # initialize
    parser = None
    if args is None:
        parser, args = parse_args()
    set_seed(args.seed)
    if os.environ.get("SGLANG_TORCH_PROFILER_DIR", "") and args.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed environment")
    print_on_rank0(f"Arguments\n{args}")
    args.dp_size = dist.get_world_size() // args.tp_size
    args.draft_accumulation_steps = (
        args.draft_global_batch_size // args.dp_size // args.draft_micro_batch_size
    )
    assert (
        args.draft_accumulation_steps * args.draft_micro_batch_size * args.dp_size
        == args.draft_global_batch_size
    ), f"draft_global_batch_size={args.draft_global_batch_size} must be divisible by dp_size={args.dp_size} and micro_batch_size={args.draft_micro_batch_size}"
    print_with_rank(
        f"draft_accumulation_steps={args.draft_global_batch_size} // {args.dp_size} // {args.draft_micro_batch_size}={args.draft_accumulation_steps}"
    )

    tracker_class = get_tracker_class(args.report_to)
    if parser:
        if tracker_class:
            tracker_class.validate_args(parser, args)
        else:
            parser.error(f"Unknown tracker: {args.report_to}")

    tracker = create_tracker(args, args.output_dir)

    # Handle draft model config
    if args.draft_model_config is None:
        # Auto-generate and save config file
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=args.cache_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        # Use provided config file
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    print_on_rank0(f"Loading target model from {args.target_model_path}...")
    if NV_JOB_CONTEXT:
        draft_model_last_checkpoint = None
        with rank_0_priority():
            ext_target_model_path = args.target_model_path + NV_EXT_SUFFIX
            print_on_rank0(f"Extracting target model to {ext_target_model_path}...")
            if dist.get_rank() == 0:
                os.mkdir(ext_target_model_path)
                subprocess.run(['tar', '-xvf', args.target_model_path, '-C', ext_target_model_path])
            args.target_model_path = ext_target_model_path

    # build target and draft model
    if args.tp_size > 1:
        # check if the target model has tp_plan
        config = AutoConfig.from_pretrained(args.target_model_path)

        if type(config) in AutoDistributedTargetModel._model_mapping:
            target_model = AutoDistributedTargetModel.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
                device="cuda",
                local_files_only=True,
            ).eval()
        else:
            target_model = AutoModelForCausalLM.from_pretrained(
                args.target_model_path,
                tp_plan="auto",
                tp_size=args.tp_size,
                torch_dtype=torch.bfloat16,
                device_mesh=get_tp_device_mesh(),
            ).eval()
    else:
        if args.is_vlm and draft_model_config.target_model_type == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            target_model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=args.target_model_path,
                    torch_dtype=torch.bfloat16,
                )
                .eval()
                .cuda()
            )
        else:
            target_model = (
                AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=args.target_model_path,
                    torch_dtype=torch.bfloat16,
                    cache_dir=args.cache_dir,
                )
                .eval()
                .cuda()
            )

    for p in target_model.parameters():
        p.requires_grad = False

    print_with_rank("Initialized target model")

    # load model with resume
    if draft_model_last_checkpoint:
        draft_model = AutoEagle3DraftModel.from_pretrained(
            draft_model_last_checkpoint,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        draft_model = AutoEagle3DraftModel.from_config(
            draft_model_config,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank("Initialized draft model")

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    else:
        processor = None

    # convert to dataloader
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    custom_preprocessing = "custom" in args.chat_template
    if custom_preprocessing:
        print_on_rank0("Using custom preprocessing implementation")
        from specforge.data.preprocessing_custom import build_eagle3_dataset
    else:
        print_on_rank0("Using base preprocessing implementation")
        from specforge.data.preprocessing import build_eagle3_dataset

    columns = ["request_id", "messages", "tools"]
    print_with_rank(f"Loading training dataset from {args.train_data_path}...")
    if args.train_data_path.startswith("yt:"):
        local_train_data_path = os.path.join(args.cache_dir, "train_dataset.tsv")
        with rank_0_priority():
            if dist.get_rank() == 0 and not os.path.exists(local_train_data_path):
                Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
                yt_pull_dataset(
                    args.train_data_path,
                    local_train_data_path,
                    token=args.yt_token,
                    columns=columns
                )
        args.train_data_path = local_train_data_path

    train_dataset = multi_load_dataset(args.train_data_path, columns=columns)
    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=args.is_vlm,
            is_preformatted=args.is_preformatted,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
            num_proc=args.build_dataset_num_proc
        )
    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.draft_micro_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=args.is_vlm,
    )
    print_with_rank(f"Initialized train dataloader with {len(train_eagle3_dataset)} samples")

    # Calculate total steps if not provided
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(
            f"Auto-calculated total_steps: {args.total_steps} (num_epochs={args.num_epochs} * steps_per_epoch={steps_per_epoch})"
        )
    else:
        print_with_rank(f"Using provided total_steps: {args.total_steps}")

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    if args.eval_data_path is not None:
        if args.eval_data_path.startswith("yt:"):
            local_eval_data_path = os.path.join(args.cache_dir, "eval_dataset.tsv")
            with rank_0_priority():
                if (dist.get_rank() == 0 and not os.path.exists(local_eval_data_path)):
                    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
                    yt_pull_dataset(
                        args.eval_data_path,
                        local_eval_data_path,
                        token=args.yt_token,
                        columns=columns
                    )
            args.eval_data_path = local_eval_data_path

        print_with_rank(f"Loading evaluation dataset from {args.eval_data_path}...")
        eval_dataset = multi_load_dataset(args.eval_data_path, columns=columns)
        eval_eagle3_dataset = build_eagle3_dataset(
            eval_dataset,
            tokenizer,
            args.chat_template,
            args.max_length,
            is_vlm=args.is_vlm,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.draft_micro_batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=args.is_vlm,
        )
        print_with_rank(f"Initialized eval dataloader with {len(eval_eagle3_dataset)} samples")

    # build Eagle3 model
    # broadcast draft model
    if args.is_vlm and draft_model_config.target_model_type == "qwen2_5_vl":
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
        eagle3_model = OnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    # eagle3_model = DDP(eagle3_model, find_unused_parameters=True)
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=[target_model],
        process_group=get_dp_group(),
    )
    print_with_rank("Initialized Eagle3 FSDP model")

    # build other components
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    # global_step
    global_step = 0
    start_epoch = 0
    if draft_model_last_checkpoint is not None:
        print_on_rank0(
            f"Resuming draft model training from checkpoint: {draft_model_last_checkpoint}"
        )
        state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state)
            start_epoch = state["epoch"] + 1
            global_step = state.get("global_step", 0)
            print_on_rank0(f"Resuming from epoch {start_epoch}")
        else:
            print_on_rank0(
                f"Warning: Checkpoint directory {draft_model_last_checkpoint} found, but training_state.pt is missing. Starting from scratch."
            )

    dist.barrier()

    last_time = time.time()

    # start running
    print_on_rank0(f"Starting training from epoch {start_epoch}")
    batch_index, log_dict = 0, defaultdict(float)

    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.module.length)]
        epoch_plosses = [[] for _ in range(eagle3_model.module.length)]

        if dist.get_rank() == 0 and args.report_to != "print":
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            batch_index += 1
            if args.profile:
                if batch_index == args.profile_start_step:
                    print_with_rank(f"Start profile")
                    torch_profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        record_shapes=args.profile_record_shapes,
                    )
                    torch_profiler.start()
                if batch_index == args.profile_start_step + args.profile_num_steps:
                    output_path = os.path.join(
                        os.environ["SGLANG_TORCH_PROFILER_DIR"],
                        f"debug_rank{torch.distributed.get_rank()}_{time.time()}.trace.json.gz",
                    )
                    print_with_rank(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)
            if args.profile_memory and batch_index == args.profile_memory_steps:
                torch.cuda.memory._dump_snapshot(os.path.join(
                    os.environ["SGLANG_TORCH_PROFILER_DIR"],
                    f"memory_rank{torch.distributed.get_rank()}_{time.time()}.pkl"
                ))
                torch.cuda.memory._record_memory_history(enabled=None)

            if args.is_vlm:
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                    pixel_values=data["pixel_values"].cuda(),
                    image_grid_thw=data["image_grid_thw"].cuda(),
                )
            else:
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                )
            acces = torch.stack(acces).cpu().tolist()

            # calculate weighted loss
            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = (
                sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
                / args.draft_accumulation_steps
            )
            ploss.backward()
            for i in range(len(plosses)):
                log_dict[f"train/ploss_{i}"] += (
                    plosses[i].item() / args.draft_accumulation_steps
                )
            for i in range(len(acces)):
                log_dict[f"train/acc_{i}"] += acces[i] / args.draft_accumulation_steps
            log_dict["train/epoch"] = epoch + global_step / steps_per_epoch
            log_dict["train/lr"] = optimizer.get_learning_rate()
            if batch_index % args.draft_accumulation_steps == 0:
                optimizer.step()
                global_step += 1
                if global_step % args.log_steps == 0:
                    tracker.log(log_dict, step=global_step)
                log_dict = defaultdict(float)

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

            if args.verbose:
                print(
                    f"[{dist.get_rank()}] time={(time.time() - last_time):.3}s shape={data['input_ids'].shape}"
                )
                last_time = time.time()

            if dist.get_rank() == 0:
                avg_loss = sum(pl.item() for pl in plosses) / len(plosses)
                avg_acc = sum(acces) / len(acces)
                try:
                    progress_bar.set_postfix(
                        {"loss": f"{avg_loss:.2f}", "acc": f"{avg_acc:.2f}"}
                    )
                except:
                    pass

            if (
                args.save_strategy == "steps" and global_step and
                (global_step % args.save_interval == 0 or global_step == args.total_steps)
            ):
                print_on_rank0(f"Saving model after step {global_step}")
                state_to_save = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": args,
                }
                state_to_save.update(optimizer.state_dict())
                output_dir = os.path.join(args.output_dir, f"step_{global_step}")
                save_model(eagle3_model, state_to_save, output_dir)

        epoch_logdict = {}
        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = (acc_i / dist.get_world_size()).item()
            epoch_logdict[f"train/epoch_acc_{i}"] = acc_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = (loss_i / dist.get_world_size()).item()
            epoch_logdict[f"train/epoch_ploss_{i}"] = loss_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )
        tracker.log(epoch_logdict, step=global_step)

        # run evaluation
        if args.eval_data_path is not None and epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_model.length)]
            eval_plosses = [[] for _ in range(eagle3_model.length)]

            if dist.get_rank() == 0 and args.report_to != "print":
                progress_bar = tqdm(
                    eval_dataloader, desc=f"Evaluation Epoch {epoch}", leave=True
                )
            else:
                print_on_rank0("Running evaluation...")
                progress_bar = eval_dataloader

            for data in eval_dataloader:
                if args.is_vlm:
                    with torch.no_grad():
                        plosses, _, acces = eagle3_model(
                            input_ids=data["input_ids"].cuda(),
                            attention_mask=data["attention_mask"].cuda(),
                            loss_mask=data["loss_mask"].cuda(),
                            pixel_values=data["pixel_values"].cuda(),
                            image_grid_thw=data["image_grid_thw"].cuda(),
                        )
                else:
                    with torch.no_grad():
                        plosses, _, acces = eagle3_model(
                            input_ids=data["input_ids"].cuda(),
                            attention_mask=data["attention_mask"].cuda(),
                            loss_mask=data["loss_mask"].cuda(),
                        )
                acces = torch.stack(acces).cpu().tolist()

                eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
                eval_plosses = [
                    eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

            # Log epoch-level evaluation metrics
            eval_logdict = {}
            for i in range(len(eval_acces)):
                acc_i = torch.tensor(eval_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = (acc_i / dist.get_world_size()).item()
                eval_logdict[f"eval/epoch_acc_{i}"] = acc_i
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(eval_plosses)):
                loss_i = torch.tensor(eval_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = (loss_i / dist.get_world_size()).item()
                eval_logdict[f"eval/epoch_ploss_{i}"] = loss_i
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )
            tracker.log(eval_logdict, step=global_step)

        if args.save_strategy != "epochs":
            continue

        if epoch % args.save_interval == 0:
            print_on_rank0(f"Saving model after epoch {epoch}")
            state_to_save = {
                "epoch": epoch,
                "global_step": global_step,
                "args": args,
            }
            state_to_save.update(optimizer.state_dict())
            output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            save_model(eagle3_model, state_to_save, output_dir)

    # Close the tracker
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
