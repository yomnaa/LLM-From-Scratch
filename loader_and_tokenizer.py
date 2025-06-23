import tiktoken
from torch.utils.data import DataLoader,Dataset
import torch
from torch import nn
from collections import Counter
# from config import *

# def create_dataloader(self,data,batch_size=32,max_context_length=8,stride=1,shuffle=False,drop_last=False,num_workers=0):
    
class Tokenizer:
    def __init__(self,model_name="gpt2"):
        self.tokenizer=tiktoken.get_encoding("gpt2")
        self.vocab_size=self.tokenizer.n_vocab

    def text_to_token_ids(self,text):
        return self.tokenizer.encode(text)

    def token_ids_to_text(self,tokens_ids):
        return self.tokenizer.decode(tokens_ids)

class GPTDataset(Dataset):
    def __init__(self,data,tokenizer,max_length,stride):
        data_tokens_ids=tokenizer.encode(data)
        self.inputs=[]
        self.targets=[]
       
        for start_idx in range(0,len(data_tokens_ids)-max_length,stride):
            end=start_idx+max_length
            self.inputs.append(torch.tensor(data_tokens_ids[start_idx:end]))
            self.targets.append(torch.tensor(data_tokens_ids[start_idx+1:end+1]))


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index],self.targets[index]

def create_data_loader(dataset,batch_size=16,shuffle=True,drop_last=True,num_workers=0):
    data_loader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return data_loader
# data=[]
# tokenizer=Tokenizer()


# dataset=GPTDataset(data,tokenizer=tokenizer.tokenizer,max_length=DatasetConfig.max_context_length,stride=DatasetConfig.stride)
# data_loader=DataLoader(dataset,batch_size=DataLoaderConfig.batch_size,shuffle=DataLoaderConfig.shuffle,drop_last=DataLoaderConfig.drop_last,num_workers=DataLoaderConfig.num_workers)

