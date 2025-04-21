
### >>>>>>>>>>>>>>>>>>>>>>>>>
### >>>>>>> reference >>>>>>>
### >>>>>>>>>>>>>>>>>>>>>>>>>
# 1. https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
# 2. https://github.com/wyf3/llm_related
# 3. https://github.com/jingtian11/EasyOffer

from copy import deepcopy
from functools import partial
from qwen_vl_utils import process_vision_info

import torch
from transformers import Trainer, AutoProcessor, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path as p
import torch.nn.functional as F

from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path as p
from transformers import HfArgumentParser, TrainingArguments


"""
You can use the two below row col to ingore the class and fun
from utils.cli import get_args
from utils.model import init_model
"""

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


def get_dummy_model(vocab_size=2000, hidden_size=16, num_hidden_layers=1, **kwargs):
    from transformers import LlamaForCausalLM, LlamaConfig
    config = LlamaConfig(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, **kwargs)
    model = LlamaForCausalLM(config)
    return model

def init_model(model_name_or_path, with_processor=False):
    try:
        model = AutoModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if with_processor:
            processor = AutoProcessor.from_pretrained(model_name_or_path)
            return model, tokenizer, processor
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model from {model_name_or_path}. Error: {e}")
        print("Maybe you just need a dummy for code. So I will init a dummy model instead.")
        model = get_dummy_model()
        return model


### >>>>>>>>>>>>>> 关于dataset >>>>>>>>>>>> 
# 可以自己定好dataset和collator进而放到trainer中，这个方法最稳了。
# 当然，也可以直接管dataset即可，然后传入tokenzier/image_process作为processer_class进而构建default_collator。
# 但这样强依赖于trainer内部逻辑，而且trainer内部的逻辑是对版本号敏感的。

# 1. 如果是offline的dpo，500question则传入1000的json，[(q1,a1_good),(q1,a1_bad),  (q2,a2_good),(q2,a2_bad)]
# 2. oneline，那就直接传prompt，然后自己生成自己打分（在train_step中自己操作rm来打分）
# 3. 同理ppo

class PromptDataset(Dataset):
    def __init__(self, data_json_file, tokenizer):
        self.data_json_file = data_json_file
        self.data_json = pd.read_json(data_json_file, lines = p(data_json_file).suffix==".jsonl")
        tokenizer.padding_side = "left"  # 虽然默认是left，但是可能用户乱动了下，在这里统一上个保险
        self.tokenizer = tokenizer
        self.samples = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in self.data_json["messages"]
        ]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        input_dict = self.tokenizer(text,
            max_length=128, padding="max_length", truncation=True,
            return_tensors="pt",
        )
        
        return input_dict


def get_mm_input_of_qwen(processor, messages, text_after_apply_chat=None, tokenizer=None):
    if text_after_apply_chat:
        assert tokenizer is not None, "tokenizer is required when text_after_apply_chat is provided"
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs
    
    
class PromptDataset_with_image_in_qwen(PromptDataset):
    def __init__(self, data_json_file, tokenizer, get_mm_input_func):
        super().__init__(data_json_file, tokenizer)
        self.get_mm_input_func = get_mm_input_func
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        messages = self.data_json["messages"][idx]
        return self.get_mm_input_func(messages, text_after_apply_chat=text)


# 上边的还是不是特别方便，我专门写一个吧。
class PromptDataset_with_image_in_qwen_dpo(PromptDataset):
    def __init__(self, data_json_file, tokenizer, get_mm_input_func):
        super().__init__(data_json_file, tokenizer)
        self.get_mm_input_func = get_mm_input_func
        
    def __len__(self):
        return super().__len__() // 2
    
    def __getitem__(self, idx):
        good_idx, bad_idx = idx * 2, idx * 2 + 1
        
        text_good, text_bad = self.samples[good_idx], self.samples[bad_idx]
        messages_good, messages_bad = self.data_json["messages"][good_idx], self.data_json["messages"][bad_idx]
        
        input_good = self.get_mm_input_func(messages_good, text_after_apply_chat=text_good)
        input_bad = self.get_mm_input_func(messages_bad, text_after_apply_chat=text_bad)
        # 可能需要stack起来，或者我后边写一个collate_fn，或者像官方一样，写一个concatenated_forward
        return [input_good, input_bad]



class DPODataCollator:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, features):
        # features是列表，每个元素是[chosen_input, rejected_input]
        chosen_batch = [f[0] for f in features]
        rejected_batch = [f[1] for f in features]
        
        # 合并chosen和rejected的输入
        batch = {}
        for key in chosen_batch[0].keys():
            values_in_batch = [torch.tensor(sample[key]) for sample in chosen_batch]
            batch[f"chosen_{key}"] = torch.cat(values_in_batch)
            
            values_in_batch = [torch.tensor(sample[key]) for sample in rejected_batch]
            batch[f"rejected_{key}"] = torch.cat(values_in_batch)

        
        # 提取labels，供之后偏移下提取log_ps
        batch["chosen_labels"] = batch["chosen_input_ids"].clone()
        batch["rejected_labels"] = batch["rejected_input_ids"].clone()
        return batch





class DPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, dpo_beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.dpo_beta = dpo_beta
        # ref model: eval and not require_grad, by this, i don't need with `with torch.no_grad()` in later
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        
    def get_batch_logps(self, logits, labels, attention_mask=None):
        # 1. Shift labels and logits: <b>123<e>, use embedding/logits of '<b>123' to decode '123<e>' in labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        # 2. Calculate per-token logps (log and gather)
        per_token_logps = torch.gather(
            shift_logits.log_softmax(2), 
            dim=2, 
            index=shift_labels.unsqueeze(2)
        ).squeeze(2)  # (batch_size,seq_len, vocab_size) -> (batch_size, seq_len)
        
        # 3. Sum logps with masking
        if attention_mask is not None:
            return (per_token_logps * shift_attention_mask).sum(-1)
        return per_token_logps.sum(-1)
    
    
    def compute_loss(self, model, inputs, **kwargs):        
        # >>> 1. concat forward in model / ref model
        # 还是放这吧，就不放datacollator里了。因为可以做各种操作，统一放这吧。从代码哲学上说，这也不属于堆叠实现的功能了，而是算法的细节。
        all_input_ids = torch.cat([inputs["chosen_input_ids"], inputs["rejected_input_ids"]], dim=0)
        all_attention_mask = torch.cat([inputs["chosen_attention_mask"], inputs["rejected_attention_mask"]], dim=0)
        if "chosen_pixel_values" in inputs and "rejected_pixel_values" in inputs:
            all_pixel_values = torch.cat([inputs["chosen_pixel_values"], inputs["rejected_pixel_values"]], dim=0)
        else:
            all_pixel_values = None
        final_concat_inputs = dict(input_ids=all_input_ids, attention_mask=all_attention_mask, pixel_values=all_pixel_values)
        
        # forward in model
        concat_outputs = model(**final_concat_inputs)
        chosen_logits = concat_outputs.logits[:inputs["chosen_input_ids"].shape[0]]
        rejected_logits = concat_outputs.logits[inputs["chosen_input_ids"].shape[0]:]
        # forward in refer
        with torch.no_grad():
            concat_outputs_of_ref = self.ref_model(**final_concat_inputs) # 保险一点吧
        chosen_logits_of_ref = concat_outputs_of_ref.logits[:inputs["chosen_input_ids"].shape[0]]
        rejected_logits_of_ref = concat_outputs_of_ref.logits[inputs["chosen_input_ids"].shape[0]:]
        
       
        # >>> 2. dpo loss
        # Calculate log probabilities
        gather_logits_of_cur_label = partial(self.get_batch_logps, labels=inputs["chosen_labels"], attention_mask=inputs["chosen_attention_mask"])
        chosen_logps, chosen_logps_of_ref = gather_logits_of_cur_label(chosen_logits), gather_logits_of_cur_label(chosen_logits_of_ref)
        gather_logits_of_cur_label = partial(self.get_batch_logps, labels=inputs["rejected_labels"], attention_mask=inputs["rejected_attention_mask"])
        rejected_logps, rejected_logps_of_ref = gather_logits_of_cur_label(rejected_logits), gather_logits_of_cur_label(rejected_logits_of_ref)
        # Compute DPO loss by two implicit rewards
        implicit_reward_chosen = chosen_logps - chosen_logps_of_ref
        implicit_reward_rejected = rejected_logps - rejected_logps_of_ref
        losses = -F.logsigmoid(self.beta * (implicit_reward_chosen - implicit_reward_rejected))
        return losses.mean()
        
        
        
        
        
def train_dpo_main():
    model_args, data_args, training_args = get_args()
    model, tokenizer, processor = init_model(model_args.model_name_or_path, with_processor=model_args.with_processor)
    ref_model = deepcopy(model)
    get_mm_input_of_qwen_25_vl = partial(get_mm_input_of_qwen, processor=processor)
    mm_dpo_dataset = PromptDataset_with_image_in_qwen_dpo(
        data_json_file=data_args.train_dataset,
        tokenizer=tokenizer,
        get_mm_input_func=get_mm_input_of_qwen_25_vl
    )
    dpo_collator = DPODataCollator(processor)
    

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=mm_dpo_dataset,
        data_collator=dpo_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    
if __name__ == '__main__':
    train_dpo_main()
    
"""
python train_dpo.py \
  --model_name_or_path /path/to/qwen-vl \
  --train_dataset /path/to/train.jsonl \
  --output_dir ./dpo_output \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 \
  --remove_unused_columns False
  
{"messages": [{"role": "user", "content": "图片里有什么动物？"}, {"role": "assistant", "content": "一只可爱的猫"}]}
{"messages": [{"role": "user", "content": "图片里有什么动物？"}, {"role": "assistant", "content": "一只可爱的猫"}]}
{"messages": [{"role": "user", "content": "图片里有什么动物？"}, {"role": "assistant", "content": "一只可爱的猫"}]}
{"messages": [{"role": "user", "content": "图片里有什么动物？"}, {"role": "assistant", "content": "一只普通的狗"}]}


之后的目标就是彻底跑起来，利用janus生成100张猫的图片，100张狗的图片。然后将label反过来，看是否可以洗脑。。
"""



