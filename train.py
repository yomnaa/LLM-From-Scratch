from llm_architecture import GPTModel
from generation import greedy_text_generation
import tiktoken
from loader_and_tokenizer import *
import os
import urllib.request
import torch
def create_split_dataloader(text_data,tokenizer,configs,batch_size=16,shuffle=True,drop_last=True,num_workers=0):
    print("batch_size",batch_size)
    dataset=GPTDataset(data=text_data,tokenizer=tokenizer,max_length=configs["context_length"],stride=configs["context_length"])
    print("dataset_length",len(dataset))
    data_loader=create_data_loader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return data_loader

def create_training_splits(text_data,tokenizer,configs):
    train_ratio=0.80
    train_idx=int(train_ratio*len(text_data))
    print("data_length",len(text_data))
    print("train_idx",train_idx)
    train_text_data=text_data[:train_idx]
    dev_idx=int(train_idx+0.50*(len(text_data)-train_idx))
    print("dev_idx",dev_idx)
    dev_text_data=text_data[train_idx:dev_idx]
    test_text_data=text_data[dev_idx:]
    print("train_length",len(train_text_data),"dev length",len(dev_text_data),"test_length",len(test_text_data))
    train_dataloader=create_split_dataloader(train_text_data,tokenizer,configs)
    dev_dataloader=create_split_dataloader(dev_text_data,tokenizer,configs,shuffle=False,drop_last=False)
    test_dataloader=create_split_dataloader(test_text_data,tokenizer,configs,shuffle=False,drop_last=False)
    return train_dataloader,dev_dataloader,test_dataloader

def iterate_over_data(dataloader):
    for x,y in dataloader:
        print(x.shape)
        print(y.shape)
        print(".........................")

def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
    logits=model(input_batch)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target=target_batch.flatten())
    return loss

def calc_loss_loader(data_loader,model,device):
    if len(data_loader)==0:
        return float("nan")
    model.eval()
    total_loss=0
    num_batches=len(data_loader)
    for input_batch,target_batch in data_loader:
        total_loss+=calc_loss_batch(input_batch,target_batch,model,device).item()
    return total_loss/num_batches

def train(model,optimizer,train_loader,val_loader,device,num_epochs,eval_freq,start_context,tokenizer):
    train_losses,val_losses=[],[]
    tokens_seen=0
    global_step=-1
    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(model)
            loss.backward()
            optimizer.step()
            
if  __name__=="__main__":
    configs={
        "vocab_size":50257,
        "emb_dim":768,
        "context_length":256,
        "n_layers":12,
        "n_heads":12,
        "drop_rate":0.1,
        "qkv_bias":False
    }

    torch.manual_seed(123)
    print("initializing model....")
    model= GPTModel(configs)
    model.eval()
    gpt_tokenizer=Tokenizer()
    # text="The boy ate"
    # length=len(text)
    # tokens=gpt_tokenizer.text_to_token_ids(text)
    # output_tokens=greedy_text_generation(model=model,
    #                        input_tokens=torch.tensor([tokens]),
    #                        max_new_tokens=10,
    #                        context_length=configs["context_length"])
    
    # print(gpt_tokenizer.token_ids_to_text(output_tokens.numpy()[0]))
    # print(gpt_tokenizer.token_ids_to_text(output_tokens.numpy()[0])[length:])
    # print(len(gpt_tokenizer.token_ids_to_text(output_tokens.numpy()[0])[length:]))



    print("loading dataset")
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    total_characters = len(text_data)
    total_tokens = len(gpt_tokenizer.text_to_token_ids(text_data))

    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    print("creating data loaders")
    train_dataloader,dev_dataloader,test_dataloader=create_training_splits(text_data,gpt_tokenizer.tokenizer,configs)
    print("train num_batches",len(train_dataloader))
    print("dev num_batches",len(dev_dataloader))
    print("test num_batches",len(test_dataloader))

    print("iterating over train")
    iterate_over_data(train_dataloader)
    print("iterating over dev")
    iterate_over_data(dev_dataloader)
    print("iterating over text")
    iterate_over_data(test_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    





    



