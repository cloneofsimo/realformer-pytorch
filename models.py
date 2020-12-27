import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    def __init__(self, emb_s = 32, head_cnt = 8, dp1 = 0.1, dp2 = 0.1):
        super().__init__()
        
        emb = emb_s*head_cnt
        self.kqv = nn.Linear(emb_s, 3*emb_s, bias = False)
        self.dp = nn.Dropout(dp1)     
        self.proj = nn.Linear(emb, emb,bias = False)
        self.head_cnt = head_cnt
        self.emb_s = emb_s
        self.ln1 = nn.LayerNorm(emb)
        self.ln2 = nn.LayerNorm(emb)
    
        self.ff = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.GELU(),
            nn.Linear(4 * emb, emb),
            nn.Dropout(dp2),
        )
        
    def mha(self, x):
        B, T, _ = x.shape
        x = x.reshape(B, T, self.head_cnt, self.emb_s)
        k, q, v = torch.split(self.kqv(x), emb_s, dim = -1) # B, T, h, emb_s
        att = F.softmax(torch.einsum('bihk,bjhk->bijh', q, k)/self.emb_s**0.5, dim = 2) #B, T, T, h sum on dim 1 = 1
        res = torch.einsum('btih,bihs->bths', att, v).reshape(B, T, -1) #B, T, h * emb_s
        return self.dp(self.proj(res))
    
    def forward(self, x): ## add & norm later.
        x = self.ln1(x + self.mha(x))
        x = self.ln2(x + self.ff(x))

        return x


class ResEncoderBlock(nn.Module):
    def __init__(self, emb_s = 32, head_cnt = 8, dp1 = 0.1, dp2 = 0.1):
        super().__init__()
        emb = emb_s * head_cnt
        self.kqv = nn.Linear(emb_s, 3*emb_s, bias = False)
        self.dp = nn.Dropout(dp1)     
        self.proj = nn.Linear(emb, emb,bias = False)
        self.head_cnt = head_cnt
        self.emb_s = emb_s
        self.ln1 = nn.LayerNorm(emb)
        self.ln2 = nn.LayerNorm(emb)
        
        self.ff = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.GELU(),
            nn.Linear(4 * emb, emb),
            nn.Dropout(dp2),
        )

    def resmha(self, x, prev = None):
        B, T, _ = x.shape
        x = x.reshape(B, T, self.head_cnt, self.emb_s)
        k, q, v = torch.split(self.kqv(x), self.emb_s, dim = -1) # B, T, h, emb_s
        if prev is not None : 
            att_score = torch.einsum('bihk,bjhk->bijh', q, k)/self.emb_s**0.5 + prev
        else:
            att_score = torch.einsum('bihk,bjhk->bijh', q, k)/self.emb_s**0.5

        prev = att_score
        att = F.softmax(prev, dim = 2) #B, T, T, h sum on dim 1 = 1
        res = torch.einsum('btih,bihs->bths', att, v).reshape(B, T, -1) #B, T, h * emb_s
        return self.dp(self.proj(res)), prev
    
    def forward(self, x, prev = None): ## add & norm later.
        rmha, prev =  self.resmha(x, prev = prev)
        x = self.ln1(x + rmha)
        x = self.ln2(x + self.ff(x))

        return x, prev
        
        
class ViT(nn.Module):
    def __init__(self, image_pix = 64, patch_pix = 4, channel_size = 3, class_cnt = 10, layer_cnt = 4, emb_s = 32, head_cnt = 8, dp1 = 0.1, dp2 = 0.1):
        super().__init__()
        emb = emb_s*head_cnt
        tokens_cnt = (image_pix//patch_pix)*(image_pix//patch_pix)
        patch_size = patch_pix*patch_pix*channel_size
        self.uf = nn.Unfold(kernel_size = [patch_pix, patch_pix], stride = [patch_pix, patch_pix])
        self.pos_emb = nn.Parameter(torch.zeros(1, tokens_cnt, emb))
        self.head = nn.Linear(emb, class_cnt)
        self.patch_emb = nn.Linear(patch_size, emb)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb))
        self.mains = nn.Sequential(*[EncoderBlock(emb_s = 32, head_cnt = 8, dp1 = 0.1, dp2 = 0.1) for _ in range(layer_cnt)])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_emb(self.uf(x).transpose(1, 2)) + self.pos_emb#(B, T, patch_size)
        x = torch.cat([self.cls_token.repeat(b, 1, 1), x], dim = 1)
        x = self.mains(x)
        x = self.head(x[:, 0, :]) #(B, class_cnt)    
        return x



class ViR(nn.Module):
    def __init__(self, image_pix = 64, patch_pix = 4, channel_size = 3, class_cnt = 10, layer_cnt = 4, emb_s = 32, head_cnt = 8, dp1 = 0.1, dp2 = 0.1):
        super().__init__()
        emb = emb_s * head_cnt
        tokens_cnt = (image_pix//patch_pix)*(image_pix//patch_pix)
        patch_size = patch_pix*patch_pix*channel_size
        self.uf = nn.Unfold(kernel_size = [patch_pix, patch_pix], stride = [patch_pix, patch_pix])
        self.pos_emb = nn.Parameter(torch.zeros(1, tokens_cnt, emb))
        self.head = nn.Linear(emb, class_cnt)
        self.patch_emb = nn.Linear(patch_size, emb)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb))
        self.mains = nn.Sequential(*[ResEncoderBlock(emb_s = 32, head_cnt = 8, dp1 = 0.1, dp2 = 0.1) for _ in range(layer_cnt)])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_emb(self.uf(x).transpose(1, 2)) + self.pos_emb#(B, T, patch_size)
        x = torch.cat([self.cls_token.repeat(b, 1, 1), x], dim = 1)
        prev = None
        for resencoder in self.mains:
            x, prev = resencoder(x, prev = prev)            
        x = self.head(x[:, 0, :]) #(B, class_cnt)    
        return x