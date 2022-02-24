import math
import torch
import numpy as np
import torch.nn as nn

#tgt_vocab_size :9


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()

    def forward(self,Q,K,V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]


        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,V)  # [batch_size, n_heads, len_q, d_v]
        return context,attn

class ABI_ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ABI_ScaledDotProductAttention, self).__init__()

    def forward(self,Q, K, V_A, V_B):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn_abi = nn.Softmax(dim=-1)(scores)
        context_a = torch.matmul(attn_abi,V_A)  # [batch_size, n_heads, len_q, d_v]
        context_b = torch.matmul(attn_abi,V_B)  # [batch_size, n_heads, len_q, d_v]

        return context_a,context_b, attn_abi


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]


        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn

class ABI_MultiHeadAttention(nn.Module):
    def __init__(self):
        super(ABI_MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V_A = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.W_V_B = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.fc_a = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.fc_b = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, q_action, k_backgroud, v_action, v_background):
        '''
        q_proposal: [batch_size, len_q, d_model]
        k_backgroud: [batch_size, len_k, d_model]
        v_proposal: [batch_size, len_v(=len_k), d_model]
        v_background: [batch_size, len_v(=len_k), d_model]
        '''
        residual_a, batch_size = q_action, q_action.size(0)
        residual_b, batch_size = k_backgroud, k_backgroud.size(0)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(q_action).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(k_backgroud).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V_A = self.W_V_A(v_action).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V_B = self.W_V_B(v_background).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context_a, context_b, attn = ABI_ScaledDotProductAttention()(Q, K, V_A, V_B)
        context_a = context_a.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        context_b = context_b.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        
        output_a = self.fc_a(context_a) # [batch_size, len_q, d_model]
        output_b = self.fc_b(context_b) # [batch_size, len_q, d_model]

        output_a = nn.LayerNorm(d_model).cuda()(output_a + residual_a)
        output_b = nn.LayerNorm(d_model).cuda()(output_b + residual_b)


        return output_a, output_b, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()

        self.enc_abi_attn = ABI_MultiHeadAttention()

        self.pos_ffn_a = PoswiseFeedForwardNet()
        self.pos_ffn_b = PoswiseFeedForwardNet()

        self.pos_ffn_aa = PoswiseFeedForwardNet()
        self.pos_ffn_bb = PoswiseFeedForwardNet()
        self.out_feat_dim = d_model

    def forward(self, action_feat, background_feat):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''

        action_query, _ = self.enc_self_attn(action_feat, action_feat, action_feat)
        key_value_bg, _ = self.enc_self_attn(background_feat, background_feat, background_feat) 
        action_query = self.pos_ffn_a(action_query) 
        key_value_bg = self.pos_ffn_b(key_value_bg)
        action_value = action_query
        output_a, output_b, attn = self.enc_abi_attn(action_query, key_value_bg, action_value, key_value_bg) 
        
        output_p = self.pos_ffn_aa(output_a) 
        output_b = self.pos_ffn_bb(output_b)

        return output_p, output_b, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb_action = PositionalEncoding(d_model)
        self.pos_emb_background = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, action_feat, background_feat):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        action_feat = self.pos_emb_action(action_feat.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        background_feat = self.pos_emb_background(background_feat.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        
        enc_self_attns = []
        for layer in self.layers:
            action_feat, background_feat, attn = layer(action_feat, background_feat)
            enc_self_attns.append(attn)
        return action_feat, background_feat, enc_self_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.in_feat_dim = 2048
        self.hidden_dim_1d = 512
        self.out_feat_dim = d_model

        self.base_module = nn.Sequential(
            nn.Conv1d(self.in_feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.out_feat_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        self.encoder = Encoder()

        self.fc_cls_a = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid()
        )
        self.fc_cls_b = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid()
        )

        self.query = nn.Embedding(256, d_model)

    def forward(self,inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        enc_inputs = self.base_module(inputs)
        enc_inputs = enc_inputs.permute(0, 2, 1)
        query_emd = ((self.query).weight).unsqueeze(0).repeat(enc_inputs.shape[0],1,1)
        enc_inputs = enc_inputs + query_emd
        action_feat = enc_inputs
        background_feat = enc_inputs
        action_feat_out, background_feat_out,_ = self.encoder(action_feat, background_feat)
        cls_a = self.fc_cls_a(action_feat_out)
        cls_b = self.fc_cls_b(background_feat_out)
        return  cls_a, cls_b, action_feat_out, background_feat_out

def get_frame_abi_model():
    global src_len, tgt_len,d_model,d_ff,d_k,n_layers,n_heads,d_v
    d_model = 400 
    d_ff = 2048 
    d_k = d_v = 64 
    n_layers = 1  
    n_heads = 8 
    model = Transformer()
    return model

# inputs = torch.randn(1,2048,256)
# model = get_frame_deep_layers()
# model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

# output = model(inputs.cuda())[3].shape
# print(output)
