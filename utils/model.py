from transformers import Trainer, AutoProcessor, AutoModel, AutoTokenizer

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

