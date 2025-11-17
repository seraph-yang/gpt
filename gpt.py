# %%
%pip install transformer_lens
%pip install einops
%pip install fancy_einsum

# %%
import einops
from fancy_einsum import einsum
from transformer_lens import EasyTransformer
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import math
from transformer_lens.utils import get_corner, gelu_new
import tqdm.auto as tqdm


# %%
# load GPT-2 small model for reference (80M params)
reference_model = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

# %%
reference_model.tokenizer

# %%
print(reference_model.to_str_tokens("seraph"))
print(reference_model.to_str_tokens("i love coding"))

# %%
print(reference_model.to_tokens("seraph")) # text to tokens
print(reference_model.to_tokens("seraph", prepend_bos=False)) # removes initial bos token
print(reference_model.to_tokens("i love coding", prepend_bos=False))

# %%
reference_text = "i love coding"
tokens = reference_model.to_tokens(reference_text)
print(tokens)
print(tokens.shape) # batch size, sequence length
print(reference_model.to_str_tokens(tokens))

# %%
logits, cache = reference_model.run_with_cache(tokens)
print(logits.shape) # batch size, sequence length, vocab size

# %%
probs = logits.softmax(dim=-1)
print(probs.shape) # batch size, sequence length, vocab size


# %%
# after each token, this is what the model thinks is next
list(zip(reference_model.to_str_tokens(reference_text), reference_model.tokenizer.batch_decode(logits.argmax(dim=-1)[0])))

# %%
# get last token logits and get highest probability token
next_token = logits[0, -1].argmax(dim=-1)
print(next_token)

# %%
next_tokens = torch.cat([tokens, torch.tensor(next_token, dtype=torch.int64)[None, None]], dim=-1)
print("new input (tokens): ", next_tokens)
print("new input (shape): ", next_tokens.shape)
print("new input (decoded): ", reference_model.tokenizer.decode(next_tokens[0]))

new_logits = reference_model(next_tokens)
print(new_logits.shape) # batch size, sequence length, vocab size



# %% [markdown]
# ---

# %%
@dataclass
class Config:
    layer_norm_eps: float = 1e-5
    init_range: float = 0.02
    n_ctx: int = 1024
    debug: bool = True
    d_model: int = 768
    d_vocab: int = 50257
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()
print(cfg)

# %% [markdown]
# ##### LayerNorm
# - make mean 0
# - normalize to have variance 1
# - scale with learned weights
# - translate with learned bias

# %%
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w = nn.Parameter(torch.ones(cfg.d_model)) # learnable weights -> by default, don't want to change the norm so set to 1
        self.b = nn.Parameter(torch.zeros(cfg.d_model)) # learnable bias -> by default, don't want to change the mean so set to 0

    def forward(self, residual): # takes in a residual vector
        if cfg.debug: print("residual shape: ", residual.shape) # batch, position, d_model
        # Calculate mean across d_model dimension and keep dimensions for broadcasting
        mean = einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        residual = residual - mean
        
        # calculate the variance (mean of squared values) and square root it; add epsilon (layer_norm_eps) to avoid division by zero
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + cfg.layer_norm_eps).sqrt()
        normalized = residual / scale

        # scale and translate
        if cfg.debug: print("normalized shape: ", normalized.shape) # batch, position, d_model
        return self.w * normalized + self.b

# %% [markdown]
# ##### Embedding

# %%
class Embedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_E = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model)) # create empty matrix with shape (d_vocab, d_model)
        nn.init.normal_(self.W_E, std=cfg.init_range) # initialize with normal distribution
    
    def forward(self, tokens): # look up embeddings for each token in the vocabulary
        if cfg.debug: print("tokens shape: ", tokens.shape) # batch, position
        embed = self.W_E[tokens, :] # batch, position, d_model

        if cfg.debug: print("embed shape: ", embed.shape) # batch, position, d_model
        return embed

random_int_test(Embedding, [2, 4])
load_gpt2_test(Embedding, reference_model.embed, tokens)

# %% [markdown]
# ##### Positional Embedding

# %%
class PosEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model)) # create empty matrix with shape (n_ctx, d_model)
        nn.init.normal_(self.W_pos, std=cfg.init_range) # initialize with normal distribution
    
    def forward(self, tokens): # look up table for positional encodings
        if cfg.debug: print("tokens shape: ", tokens.shape) # batch, position
        pos_embed = self.W_pos[:tokens.size(1), :] # position, d_model
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))

        if cfg.debug: print("pos_embed shape: ", pos_embed.shape) # batch, position, d_model
        return pos_embed

random_int_test(PosEmbedding, [2, 4])
load_gpt2_test(PosEmbedding, reference_model.pos_embed, tokens)

# %% [markdown]
# ##### Attention
# - produce attention pattern -> for each destination token, probability distribution over previous tokens
#     - linear map from input -> query, key
#     - dot product every *pair* of queries and keys to get attention scores
#     - scale and mask to make it causal
#     - softmax row-wise to get a probability distribution
# - move information from source tokens to destination tokens (apply a linear map)
#     - linear map from input -> value
#     - mix along key position with attention pattern to get a mixed value
#     - map to output

# %%
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_Q, std=cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))

        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_K, std=cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))

        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_V, std=cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))

        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        nn.init.normal_(self.W_O, std=cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32))

    def forward(self, normalized_resid_pre):
        if cfg.debug: print("normalized_resid_pre shape: ", normalized_resid_pre.shape) # batch, position, d_model

        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K

        # attention scores: dot product of q and k -> bilinear form of inputs
        attention_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attention_scores = attention_scores / math.sqrt(self.cfg.d_head)

        # apply a mask to the attention scores
        attention_scores = self.apply_causal_mask(attention_scores)
        attention = attention_scores.softmax(dim=-1) # [batch, n_heads, query_pos, key_pos]
        
        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", attention, v)
        attention_output = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z, self.W_O) + self.b_O
        
        return attention_output

    def apply_causal_mask(self, attention_scores):
        # make a upper triangular matrix of ones and set to true (booleans)
        mask = torch.triu(torch.ones(attention_scores.size(-2), attention_scores.size(-1), device=attention_scores.device), diagonal=1).bool()
        attention_scores.masked_fill_(mask, self.IGNORE) # edit in place with IGNORE

        return attention_scores
    
random_float_test(Attention, [2, 4, 768])
load_gpt2_test(Attention, reference_model.blocks[0].attn, cache["blocks.0.ln1.hook_normalized"])


# %% [markdown]
# ##### MLP

# %%
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        nn.init.normal_(self.W_in, std=cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))

        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))
        nn.init.normal_(self.W_out, std=cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, normalized_resid_mid):
        if cfg.debug: print("normalized_resid_mid shape: ", normalized_resid_mid.shape) # batch, position, d_model

        # preactivation
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu_new(pre)
        
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out

random_float_test(MLP, [2, 4, 768])
load_gpt2_test(MLP, reference_model.blocks[0].mlp, cache["blocks.0.ln2.hook_normalized"])

# %% [markdown]
# ##### Transformer Block

# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        normalized_resid_pre = self.ln1(resid_pre)
        attention_out = self.attn(normalized_resid_pre)
        resid_mid = resid_pre + attention_out
        
        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post

random_float_test(TransformerBlock, [2, 4, 768])
load_gpt2_test(TransformerBlock, reference_model.blocks[0], cache["resid_pre", 0])

# %% [markdown]
# ##### Unembedding

# %%
class Unembedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.W_U = nn.Parameter(torch.empty(cfg.d_model, cfg.d_vocab))
        nn.init.normal_(self.W_U, std=cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros(cfg.d_vocab), requires_grad=False)

    def forward(self, normalized_resid_post):
        if cfg.debug: print("normalized_resid_post shape: ", normalized_resid_post.shape) # batch, position, d_model

        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_post, self.W_U) + self.b_U
        return logits
        

# %% [markdown]
# ##### Full Transformer

# %%
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed = Embedding(cfg)
        self.pos_embed = PosEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembedding(cfg)

    def forward(self, tokens):
        # tokens [batch, position]
        embed = self.embed(tokens) # [batch, position, d_model]
        pos_embed = self.pos_embed(tokens) # [batch, position, d_model]

        residual = embed + pos_embed
        for block in self.blocks:
            residual = block(residual)
        
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)

        return logits

random_int_test(Transformer, [2, 4])
load_gpt2_test(Transformer, reference_model, tokens)

            

# %%
nanogpt = Transformer(Config(debug=False))
nanogpt.load_state_dict(reference_model.state_dict(), strict=False)
nanogpt = nanogpt.to(next(reference_model.parameters()).device)

# %%
test_input = "Y Combinator (YC) initiated operations with concurrent programs in Cambridge, Massachusetts, and Mountain View, California. However, operational complexities arising from managing two programs prompted a consolidation in January 2009, resulting in the closing of the Cambridge program and the centralization of activities in Silicon Valley."
test_tokens = reference_model.to_tokens(test_input)
test_logits = nanogpt(test_tokens)

# %%
def lm_cross_entropy_loss(logits, tokens):
    # measure next token loss
    # logits: [batch, seq_len, vocab_size]
    # tokens: [batch, seq_len]
    # for each position i, we predict token i+1
    # so we use logits at position i to predict token at position i+1
    
    log_probs = logits.log_softmax(dim=-1)
    
    # Ensure sequence lengths match
    seq_len = min(logits.shape[1], tokens.shape[1])
    
    # logits[:, :-1] predicts tokens at positions 1 to seq_len-1
    # tokens[:, 1:] are the actual tokens at positions 1 to seq_len-1
    pred_log_probs = log_probs[:, :seq_len-1].gather(
        dim=-1, 
        index=tokens[:, 1:seq_len].unsqueeze(-1)).squeeze(-1)

    return -pred_log_probs.mean()

lm_cross_entropy_loss(test_logits, test_tokens)

# %%
test_string = "Y Combinator (YC) initiated operations with concurrent programs in Cambridge, Massachusetts, and Mountain View, California. However, operational complexities arising from managing two programs prompted a consolidation in January 2009, resulting in the closing of the Cambridge program and the centralization of activities in Silicon Valley."
for i in tqdm.tqdm(range(100)):
    test_tokens = reference_model.to_tokens(test_string)
    demo_logits = nanogpt(test_tokens)
    test_string += reference_model.tokenizer.decode(demo_logits[-1, -1].argmax())
print(test_string)


# %% [markdown]
# ##### Tests

# %%
def random_float_test(class_name, shape):
    cfg = Config(debug=True)
    layer = class_name(cfg)

    random_input = torch.randn(shape) # standard gaussian
    print("input shape: ", random_input.shape)

    output = layer(random_input)
    print("output shape: ", output.shape)

    print()
    return output

def random_int_test(class_name, shape):
    cfg = Config(debug=True)
    layer = class_name(cfg)

    random_input = torch.randint(100, 1000, shape)
    print("input shape:", random_input.shape)

    output = layer(random_input)
    print("output shape:", output.shape)
    print()

    return output

def load_gpt2_test(class_name, gpt2_layer, input_name, cache_dict=cache.cache_dict):
    cfg = Config(debug=True)
    layer = class_name(cfg)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    
    # allow inputs of strings or tensors
    if isinstance(input_name, str): 
        reference_input = cache_dict[input_name]
    else:
        reference_input = input_name
    # move layer to the same device as the input
    device = reference_input.device
    layer = layer.to(device)
    print("input shape:", reference_input.shape)
    output = layer(reference_input)
    print("output shape:", output.shape)

    # gpt2 attention needs 3 inputs
    if class_name.__name__ == "Attention":
        reference_output = gpt2_layer(reference_input, reference_input, reference_input)
    else:
        reference_output = gpt2_layer(reference_input)
    print("reference output shape:", reference_output.shape)

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct")
    return output

# %%
random_float_test(LayerNorm, [2, 4, 768])


