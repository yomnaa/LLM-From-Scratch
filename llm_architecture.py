from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
    
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.shift=nn.Parameter(torch.zeros(emb_dim))
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.eps=1e-5

    def forward(self,x_input:torch.Tensor):
        mean=x_input.mean(dim=-1,keepdim=True)
        var=x_input.var(dim=-1,keepdim=True,unbiased=False)
        normalized_x_input=(x_input-mean)/(torch.sqrt(var)+self.eps)
        return self.scale*normalized_x_input+self.shift

class GLEU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x_input):
        return 0.5 * x_input * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x_input + 0.044715 * torch.pow(x_input, 3))
        ))
'''
    def __init__(self):
        super().__init__()

    def forward(self,x_input):
        pass
'''
class FeedForwardLayer(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(emb_dim,4*emb_dim),
                                  GLEU(),
                                  nn.Linear(4*emb_dim,emb_dim))

    def forward(self,x_input):
        return self.layers(x_input)
    

class TransformerLayer(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.pre_attention_norm=LayerNorm(configs["emb_dim"])
        self.pre_feedforward_norm=LayerNorm(configs["emb_dim"])
        self.attention=MultiHeadAttention(
            d_in=configs["emb_dim"],
            d_out=configs["emb_dim"],
            context_length=configs["context_length"],
            dropout=configs["drop_rate"],
            num_heads=configs["n_heads"],
            qkv_bias=configs["qkv_bias"]
        )
        self.feed_forward=FeedForwardLayer(configs["emb_dim"])
        self.drop_shortcut=nn.Dropout(configs["drop_rate"])

    def forward(self,x_input):
        shortcut=x_input
        output=self.pre_attention_norm(x_input)
        output=self.attention(output)
        output=self.drop_shortcut(output)
        output=shortcut+output
        
        shortcut=output
        output=self.pre_feedforward_norm(output)
        output=self.feed_forward(output)
        output=self.drop_shortcut(output)
        output=output+shortcut
        return output

    
class GPTModel(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.token_embeddings=nn.Embedding(configs["vocab_size"],configs["emb_dim"])
        self.pos_embeddings=nn.Embedding(configs["context_length"],configs["emb_dim"])
        self.emb_dropout=nn.Dropout(configs["drop_rate"])
        self.transformer_layers=nn.Sequential(*[
            TransformerLayer(configs) for _ in range(configs["n_layers"])
        ])
        self.final_norm=LayerNorm(configs["emb_dim"])
        self.logits_head=nn.Linear(configs["emb_dim"],configs["vocab_size"],bias=False)

    def forward(self,x_input):
        batch_size,n_tokens=x_input.shape
        token_embeddings=self.token_embeddings(x_input)
        pos_embeddings=self.pos_embeddings(torch.arange(n_tokens,device=x_input.device))
        output=token_embeddings+pos_embeddings
        output=self.emb_dropout(output)
        output=self.transformer_layers(output)
        output=self.final_norm(output)
        logits=self.logits_head(output)
        return logits


from generation import greedy_text_generation
from loader_and_tokenizer import Tokenizer
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
    model= GPTModel(configs)
    model.eval()
    gpt_tokenizer=Tokenizer()
    text="The boy ate"
    length=len(text)
    tokens=gpt_tokenizer.text_to_token_ids(text)
    output_tokens=greedy_text_generation(model=model,
                           input_tokens=torch.tensor([tokens]),
                           max_new_tokens=10,
                           context_length=configs["context_length"])
    
    print(gpt_tokenizer.token_ids_to_text(output_tokens.numpy()[0]))
    print(gpt_tokenizer.token_ids_to_text(output_tokens.numpy()[0])[length:])
    print(len(gpt_tokenizer.token_ids_to_text(output_tokens.numpy()[0])[length:]))


