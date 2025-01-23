import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

def build_model(model_name): 
    if model_name == 'GPT2': 
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
    elif model_name == 'LLaMA': 
        # Replace with location of LLaMA model weights
        model_dir = "meta-llama/Llama-2-7b-chat-hf"
        
        access_token='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'
        #model_dir = "/home/arian/Documents/Projects/llama/llama-2-7b-hf"

        tokenizer = LlamaTokenizer.from_pretrained(model_dir, token-access_token)
        model = LlamaForCausalLM.from_pretrained(model_dir, token-access_token)
    else:
        print("Model doesn't exist: "+str(model_name))
    
    return model, tokenizer 

def check_equality(org_model, new_model): 
    # Models must be of same size
    new_model_dict = dict(new_model.named_parameters())
    for name, parameter in org_model.named_parameters(): 
        param_new = new_model_dict[name]
        if not torch.equal(parameter.data, param_new.data): 
            print("Inequality at param: {}".format(name))