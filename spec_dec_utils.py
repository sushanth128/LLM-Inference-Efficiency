import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np

def sample(logits, num_samples=1): 
    probabilities = F.softmax(logits, dim=-1)
    samples = torch.multinomial(probabilities, num_samples=num_samples)

    return samples 

def sample_from_diff(target_logits, draft_logits, num_samples=1): 
    target_probs = F.softmax(target_logits, dim=-1)
    draft_probs = F.softmax(draft_logits, dim=-1)
    diff = nn.ReLU()(target_probs - draft_probs)
    # print(diff)
    # print(diff.shape)
    samples = torch.multinomial(diff, num_samples=num_samples)

    return samples 

def speculative_decoding(target_model, draft_model, tokenizer, prompt, max_seq_len, num_tokens): 
    prefix_ids = tokenizer.encode(prompt, return_tensors="pt")#.cuda() 
    # print(prefix_ids.shape)

    # Length of the current sequence 
    seq_len = prefix_ids.shape[1]

    # Continue generation until max_seq_len is exceeded 
    while seq_len < max_seq_len: 

        # Sample num_tokens from draft model and add them to the prefix 
        for i in range(num_tokens): 
            draft_logits = draft_model(prefix_ids).logits 

            # Get logits from last token and sample next token 
            last_tok_logits = draft_logits[:, -1, :]
            tok = sample(last_tok_logits)#.cuda() 

            # Add to prefix 
            prefix_ids = torch.cat((prefix_ids, tok), dim=1)

        # Generate logits for prefix using target model 
        target_logits = target_model(prefix_ids).logits 

        # Verify tokens 
        num_accepted_tokens = 0 
        for i in range(num_tokens): 
            r = np.random.uniform(0, 1)
            draft_logit = draft_logits[:, seq_len + i - 1, :]
            target_logit = target_logits[:, seq_len + i - 1, :]

            draft_probs = F.softmax(draft_logit, dim=-1)
            target_probs = F.softmax(target_logit, dim=-1)

            tok = (prefix_ids[0, seq_len + i - 1]).item()

            draft_prob = draft_probs[0, tok].item()
            target_prob = target_probs[0, tok].item()

            if draft_prob != 0: 
                threshold = min(1, target_prob / draft_prob)
            else: 
                threshold = 1

            # Token is accepted 
            if r < threshold: 
                # print("Token accepted")
                num_accepted_tokens += 1 
            # Token is rejected 
            else: 
                # print("Token rejected")
                seq_len += num_accepted_tokens
                prefix_ids = prefix_ids[:, :seq_len]
                fin_target_logits = target_model(prefix_ids).logits 
                fin_draft_logits = draft_model(prefix_ids).logits 
                fin_tok = sample_from_diff(fin_target_logits[:, -1, :], fin_draft_logits[:, -1, :], num_samples=1)
                prefix_ids = torch.cat((prefix_ids, fin_tok), dim=1)
                seq_len += 1 
                break 

        # All tokens were accepted 
        if num_accepted_tokens == num_tokens: 
            seq_len += num_tokens
            fin_target_logits = target_model(prefix_ids).logits 
            fin_tok = sample(fin_target_logits[:, -1, :], num_samples=1)
            prefix_ids = torch.cat((prefix_ids, fin_tok), dim=1)
            seq_len += 1 
        
        # print(prefix_ids)
        # print(seq_len)
    
    return prefix_ids
