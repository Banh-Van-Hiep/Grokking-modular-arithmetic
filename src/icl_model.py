import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    a,b=x.chunk(2,dim=-1)
    return torch.cat([-b,a],dim=-1)

class RoPECache(nn.Module):
    def __init__(self,head_dim,max_seq,theta):
        super().__init__()
        inv=1/(theta**(torch.arange(0,head_dim,2).float()/head_dim))
        t=torch.arange(max_seq).float()
        emb=torch.cat([torch.einsum('i,j->ij',t,inv)]*2,-1)
        self.register_buffer('cos',emb.cos()[None,None,:,:],persistent=False)
        self.register_buffer('sin',emb.sin()[None,None,:,:],persistent=False)

    def forward(self,q,k,T):
        c,s=self.cos[:,:,:T,:],self.sin[:,:,:T,:]
        q=q*c+rotate_half(q)*s
        k=k*c+rotate_half(k)*s
        return q,k

class Attention(nn.Module):
    def __init__(self,d,h,rope):
        super().__init__()
        self.h=h
        self.dh=d//h
        self.qkv=nn.Linear(d,3*d,bias=False)
        self.proj=nn.Linear(d,d,bias=False)
        self.rope=rope

    def forward(self,x):
        B,T,D=x.shape
        q,k,v=self.qkv(x).view(B,T,3,self.h,self.dh).permute(2,0,3,1,4)
        q,k=self.rope(q,k,T)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.proj(o.transpose(1,2).reshape(B,T,D))

class MLP(nn.Module):
    def __init__(self,d,inner):
        super().__init__()
        self.fc1=nn.Linear(d,inner,bias=False)
        self.fc2=nn.Linear(inner,d,bias=False)

    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self,d,h,inner,rope):
        super().__init__()
        self.ln1=nn.LayerNorm(d)
        self.attn=Attention(d,h,rope)
        self.ln2=nn.LayerNorm(d)
        self.mlp=MLP(d,inner)

    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        return x+self.mlp(self.ln2(x))

class RoPETransformer(nn.Module):
    def __init__(self,vocab,d,n_layer,n_head,inner,max_seq,theta):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab,d)
        rope=RoPECache(d//n_head,max_seq,theta)
        self.blocks=nn.ModuleList([Block(d,n_head,inner,rope) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,vocab,bias=False)
        self.head.weight=self.tok_emb.weight
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.mlp.fc2.weight,0,0.02/math.sqrt(2*n_layer))

    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,0,0.02)

    def forward(self,idx):
        x=self.tok_emb(idx)
        for b in self.blocks:
            x=b(x)
        return self.head(self.ln_f(x))