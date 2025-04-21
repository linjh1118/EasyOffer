import transformers
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path as p
from transformers import HfArgumentParser, TrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/workspace/official_pretrains/hf_home/Qwen2.5-VL-3B-Instruct")
    with_processor: bool = field(default=True, metadata={"help": "Whether to load the processor."})

@dataclass
class DataArguments:
    train_dataset: str = field(default='dataset/train', metadata={"help": "Path to the training data."})
    eval_dataset: str = field(default='dataset/eval', metadata={"help": "Path to the evaluation data."})

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="output")


def get_args(config: Optional[Union[Dict, str]] = None, model_args_cls=None, data_args_cls=None, training_args_cls=None, total_args_cls=None):
    #### 1. Define the argument parser
    if total_args_cls:
        # only one class
        parser = HfArgumentParser(total_args_cls)
    elif model_args_cls and data_args_cls and training_args_cls:
        # self-define three classes
        parser = HfArgumentParser((model_args_cls, data_args_cls, training_args_cls))
    else:
        # default three classes
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    #### 2. Parse the arguments
    # from command line arguments
    if config is None:
        return parser.parse_args_into_dataclasses()
    # from dict
    if isinstance(config, dict):
        return parser.parse_dict(args=config, allow_extra_keys=False)
    # from json / yaml file
    if isinstance(config, str) and (config_path := p(config)).exists():
        assert config_path.suffix in [".json", ".yaml", ".yml"], f"Unsupported file format: {config_path.suffix}. Use JSON/YAML or command line arguments."
        if config_path.suffix == ".json":
            return parser.parse_json_file(json_file=config_path)
        elif config_path.suffix in [".yaml", ".yml"]:
            return parser.parse_yaml_file(yaml_file=config_path)
    raise ValueError(f"Unsupported config type: {type(config)}. Use a dictionary, JSON file, YAML file, or command line arguments.")

def test_case():
    config_dict = {
        "model_name_or_path": "Qwen/Qwen1.5-14B-Chat",
        "train_dataset": "dataset/train",
        "eval_dataset": "dataset/eval",
        "output_dir": "output"
    }
    model_args, data_args, training_args = get_args(config=config_dict)
    print(model_args)
    print(data_args)
    
if __name__ == "__main__":
    test_case()