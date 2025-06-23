import torch
def greedy_text_generation(model,input_tokens,max_new_tokens,context_length):
    model.eval()
    for _ in range(max_new_tokens):
        current_input_tokens=input_tokens[:,-context_length:]
        logits=model(current_input_tokens)
        last_token_logits=logits[:,-1,:]
        next_token_idx=torch.argmax(last_token_logits,dim=-1,keepdim=True)
        input_tokens=torch.cat((input_tokens,next_token_idx),dim=-1)

    return input_tokens

