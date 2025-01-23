import torch 
from transformers import AutoConfig, AutoModelForCausalLM

def calc_transformer_blocks(target_model, num_layers): 
    target_blocks = len(target_model.transformer.h)
    selected_blocks = [i for i in range(0, target_blocks, target_blocks // num_layers)]
    
    return selected_blocks

def copy_remaining_layers(target_model, draft_model): 
    target_model_dict = dict(target_model.named_parameters())
    for name, parameter in draft_model.named_parameters():
        if 'transformer.h' not in name: 
            parameter.data = target_model_dict[name].data 

def compress_model(target_model, model_name, selected_blocks): 
    if model_name == 'GPT2': 
        config = AutoConfig.from_pretrained('gpt2')
        config.n_layer = len(selected_blocks)
        draft_model = AutoModelForCausalLM.from_config(config)
    elif model_name == 'LLaMA': 
        pass 
    
    for i, block_idx in enumerate(selected_blocks): 
        draft_model.transformer.h[i].load_state_dict(target_model.transformer.h[block_idx].state_dict())
    
    copy_remaining_layers(target_model, draft_model)
    
    return draft_model 