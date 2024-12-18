import torch as tc
import MultiHead as mh
import FeedForward as FFN 

class DecoderLayer(tc.nn.Module):
    def __init__(self, h, d_model, ffn_dim):
        super().__init__()
        self.h = h
        self.d = d_model

        # embedding is expected to have dims (d_batch, length, d_model) 
        self.W_q_masked = tc.nn.Linear(d_model, d_model)
        self.W_k_masked = tc.nn.Linear(d_model, d_model)
        self.W_v_masked = tc.nn.Linear(d_model, d_model)

        self.MaskedMultiHead = mh.MultiHead( h, d_model)
        self.multiHead = mh.MultiHead( h, d_model)
        self.feedForward = FFN.FeedForward(d_model, ffn_dim)

        self.norm1 = tc.nn.LayerNorm(d_model)
        self.norm2 = tc.nn.LayerNorm(d_model)
        self.norm3 = tc.nn.LayerNorm(d_model)
    
    
    def forward(self, X, encoder_output):
        Q = self.W_q_masked(X)
        K = self.W_k_masked(X)
        V = self.W_v_masked(X) 
        
        seq_len = X.size(1)
        LA_mask = tc.triu(tc.ones((seq_len, seq_len)), diagonal=1).bool().to(X.device)

        X_1 = self.MaskedMultiHead(V, K, Q, LA_mask) #(self, V, K, Q, flag="none")
        X_1 = self.norm1(X_1 + X)

        X_2 = self.multiHead(encoder_output, encoder_output, X_1)
        X_2 = self.norm2(X_2 + X_1)

        X_3 = self.feedForward(X_2)
        X_3 = self.norm3(X_3 + X_2)
        return X_3
    
    
