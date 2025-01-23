import torch 
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig 
import copy 

def convert_fp16(model): 
    copied = copy.deepcopy(model)
    for param in copied.parameters():
        param.data = param.data.half()
    return model

def quantize_gptq(model, tokenizer, bits): 
    quantizer = GPTQQuantizer(bits=bits, dataset='c4')
    quantized_model = quantizer.quantize_model(model, tokenizer)

    return quantized_model 

def quantize_nf4(model, bits): 
    model_id = model.config._name_or_path

    if bits == 4: 
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    elif bits == 8: 
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    quantized_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    return quantized_model

def quantize(model, quantization_method, tokenizer, bits): 
    if bits == 16: 
        quantized_model = convert_fp16(model)
    else: 
        if quantization_method == 'GPTQ': 
            quantized_model = quantize_gptq(model, tokenizer, bits)
        elif quantization_method == 'NF4': 
            quantized_model = quantize_nf4(model, bits)
    
    return quantized_model 