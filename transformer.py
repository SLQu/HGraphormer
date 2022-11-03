import torch
import numpy as np

def laplance(H,device):
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = torch.ones(n_edge,dtype=torch.float32).to(device)
    # the degree of the node
    DV = torch.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = torch.sum(H, axis=0)
    invDE = torch.diag(DE ** -1)
    DV2 = torch.diag(DV ** -0.5)
    DV2[0,0] = 1
    W = torch.diag(W)
    HT = H.T 
    DV2_H = torch.matmul(DV2,H)
    DV2_H_W = torch.matmul(DV2_H,W)
    DV2_H_W_invDE = torch.matmul(DV2_H_W,invDE)
    DV2_H_W_invDE_HT = torch.matmul(DV2_H_W_invDE,HT)
    G = torch.matmul(DV2_H_W_invDE_HT,DV2)
    return G

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self,d_k,gamma):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.gamma = gamma

    def forward(self, Q, K, V, attn_mask,visible_M=None):
        # scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, nhead, seq_len, seq_len]

        scores = torch.matmul(Q, K.transpose(-1, -2)) 
        scores = scores/ np.sqrt(self.d_k) # scores : [ nhead, seq_len, seq_len]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn*self.gamma+visible_M*(1-self.gamma), V),attn
        return context
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention, self).__init__()

        self.nhid = args.nhid
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.nhead = args.nhead
        self.residual = args.residual

        self.W_Q = torch.nn.Linear(self.nhid, self.d_k * self.nhead)
        self.W_K = torch.nn.Linear(self.nhid, self.d_k * self.nhead)
        self.W_V = torch.nn.Linear(self.nhid, self.d_v * self.nhead)
        self.out_Linear = torch.nn.Linear(self.nhead * self.d_v, self.nhid)
        self.out_LayerNorm = torch.nn.LayerNorm(self.nhid)
        
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k,args.gamma)

    def forward(self, Q, K, V, attn_mask,visible_M = None):
        # q: [batch_size, seq_len, nhid], k: [batch_size, seq_len, nhid], v: [batch_size, seq_len, nhid]
        residual = V
        q_s = self.W_Q(Q).view(-1, self.nhead, self.d_k).transpose(0,1)  # q_s: [nhead, seq_len, d_k]
        k_s = self.W_K(K).view(-1, self.nhead, self.d_k).transpose(0,1)  # k_s: [nhead, seq_len, d_k]
        v_s = self.W_V(V).view(-1, self.nhead, self.d_v).transpose(0,1)  # v_s: [nhead, seq_len, d_v]
        
        if not visible_M==None:
            visible_M = visible_M.unsqueeze(0).repeat(self.nhead,1,1)
        attn_mask = attn_mask.unsqueeze(0).repeat(self.nhead,1,1) # attn_mask : [batch_size, nhead, seq_len, seq_len]
        
        context,attn = self.ScaledDotProductAttention(q_s, k_s, v_s, attn_mask,visible_M)
        context = context.transpose(0, 1).contiguous().view(-1, self.nhead * self.d_v) # context: 【seq_len, nhead * d_v]
        output = self.out_Linear(context)
        if self.residual:
            return self.out_LayerNorm(output + residual)# output: [batch_size, seq_len, nhid]
        else:  
            return self.out_LayerNorm(output) # output: [batch_size, seq_len, nhid]

# class PoswiseFeedForwardNet(torch.nn.Module):
#     def __init__(self,args):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.fc1 = nn.Linear(args.nhid, args.nhid)
#         self.fc2 = nn.Linear(args.nhid,args.nhid )
        

#     def forward(self, x):
#         return self.fc2(gelu(self.fc1(x)))

       
       

class EncoderLayer(torch.nn.Module):
    def __init__(self,args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        # self.pos_ffn = PoswiseFeedForwardNet(args)
        self.dropout = torch.nn.Dropout(args.dropout)
        # if args.act == 'gelu':
        #     self.activate = gelu
        # elif args.act == 'ReLU':
        #     self.activate = nn.ReLU()
        # elif args.act == 'PReLU':
        self.activate = torch.nn.PReLU()
        

    def forward(self,enc_inputs, enc_self_attn_mask,visible_M=None):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask,visible_M) # enc_inputs to same Q,K,V

        enc_outputs = self.activate(enc_outputs)
        enc_outputs = self.dropout(enc_outputs)
        return enc_outputs
   
class HGraphormer(torch.nn.Module):
    def __init__(self,args):
        super(HGraphormer, self).__init__()

        self.vocab_size, self.nhid, self.n_segments, self.maxlen = args.node_num+1,args.nhid,args.edge_num,args.maxlen
        self.nlayer = args.nlayer
        self.nclass = args.nclass
        self.layers = torch.nn.ModuleList([EncoderLayer(args) for _ in range(self.nlayer)])

        self.feature2hid = torch.nn.Linear(args.nfeat, self.nhid)
        self.classifier = torch.nn.Linear(args.nhid, self.nclass)
        self.norm = torch.nn.LayerNorm(self.nhid)
        
    def forward(self, device, _X0_, laplanceM,sub_full_idx,node2edge,edge2node):
        
        maxlen = laplanceM.shape[0]
        laplanceM = laplance(laplanceM,device) 
        X =self.feature2hid(_X0_)
        
        enc_self_attn_mask = torch.zeros(size=[maxlen,maxlen]).eq(1).to(device) # [] maxlen, maxlen]
        output = X.clone()
        del X
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask, laplanceM)
            
        logist = self.classifier(output)
        log_softmax_ = torch.nn.functional.log_softmax(logist, dim=1)
        return log_softmax_
   