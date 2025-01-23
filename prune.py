import torch 
import numpy as np
import copy

def get_params_per_layer(model, structured, block_size=None): 
    num_weights = [] 
    cumulative = 0 
    for name, parameter in model.named_parameters(): 
        # print(parameter.dim())
        num_weights.append(cumulative)
        if structured:
            if parameter.dim() == 1: 
                blocks = parameter.shape[0] // block_size 
            else: 
                blocks = (parameter.shape[0] // block_size) * (parameter.shape[1] // block_size)
            cumulative += blocks 
        else: 
            cumulative += torch.numel(parameter)
    
    num_weights.append(cumulative)
    return num_weights

def init_scores(model, prune_method, structured, block_size=None): 
    scores = torch.Tensor([]).cuda()
    params_per_layer = get_params_per_layer(model, structured, block_size=block_size)

    # Generate initial scores randomly
    if prune_method == "random":
        scores = torch.rand(params_per_layer[-1])
    
    # Generate initial scores based on weight
    elif prune_method == "l1": 
        for name, parameter in model.named_parameters(): 
            layer_weights = copy.deepcopy(parameter).cuda()
            layer_weights = torch.abs(layer_weights)
            if structured: 
                if parameter.dim() == 1: 
                    num_blocks = layer_weights.size(0) // block_size
                    reshaped_tensor = layer_weights[:num_blocks * block_size].view(num_blocks, block_size)
                    l1_norms = reshaped_tensor.sum(dim=1)
                    scores = torch.cat((scores, l1_norms))
                else: 
                    num_blocks_row = layer_weights.size(0) // block_size
                    num_blocks_col = layer_weights.size(1) // block_size
                    reshaped_tensor = layer_weights[:num_blocks_row * block_size, :num_blocks_col * block_size].view(num_blocks_row, block_size, num_blocks_col, block_size)
                    l1_norms = reshaped_tensor.sum(dim=(1, 3)).flatten() 
                    scores = torch.cat((scores, l1_norms))
            else: 
                scores = torch.cat((scores, layer_weights.flatten()))

    else: 
        raise Exception("Score initialization not valid")
    
    return scores

def get_topk_mask(importances, density): 
    mask = torch.zeros_like(importances).bool()
    k = int(importances.numel() * density)
    _, keep_idx = torch.topk(importances, k=k)
    mask[keep_idx] = 1
    return mask

def apply_topk_pruning(model, prune_method, density, structured, block_size=None):
    curr_layer = 0 
    copied = copy.deepcopy(model)
    params_per_layer = get_params_per_layer(copied, structured, block_size=block_size)
    importances = init_scores(copied, prune_method, structured, block_size=block_size)

    for _, parameter in copied.named_parameters():
        layer_importances = importances[params_per_layer[curr_layer]:params_per_layer[curr_layer + 1]]
        flat_mask = get_topk_mask(layer_importances, density)
        if structured: 
            if parameter.dim() == 1: 
                mask = flat_mask.repeat_interleave(block_size).cuda()
            else: 
                num_blocks_row = parameter.data.size(0) // block_size
                num_blocks_col = parameter.data.size(1) // block_size 
                reshaped_mask = flat_mask.reshape((num_blocks_row, num_blocks_col))
                mask = reshaped_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1).cuda()
        else: 
            mask = torch.reshape(flat_mask, parameter.data.size()).cuda() 

        if mask.shape != parameter.data.shape: 
            padded_tensor = torch.ones(parameter.data.shape)
            if parameter.dim() == 1: 
                padded_tensor[:mask.size(0)] = mask
            else: 
                padded_tensor[:mask.size(0), :mask.size(1)] = mask 
            mask = padded_tensor

        with torch.no_grad(): 
            parameter.data[mask == 0] = 0
        
        curr_layer += 1 
    
    return copied