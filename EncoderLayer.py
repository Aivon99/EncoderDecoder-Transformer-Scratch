import torch as tc
import MultiHead as mh
import FeedForward as FFN


class EncoderLayer(tc.nn.Module):
    
    def __init__(self, h, d_model, ffn_dim):
        super().__init__()
        self.h = h
        self.d = d_model

        # embedding is expected to have dims (d_batch, length, d_model) 
        self.W_q = tc.nn.Linear(d_model, d_model)
        self.W_k = tc.nn.Linear(d_model, d_model)
        self.W_v = tc.nn.Linear(d_model, d_model)

        self.multiHead = mh.MultiHead( h, d_model)
        self.feedForward = FFN.FeedForward(d_model, ffn_dim)

        self.norm1 = tc.nn.LayerNorm(d_model)
        self.norm2 = tc.nn.LayerNorm(d_model)

    def forward(self, X):
        

        Q =self.W_q(X)
        V =self.W_v(X)
        K =self.W_k(X)

        add = self.multiHead(Q, V, K)
         
        X += add   
        X = self.norm1(X)

        X_f = self.feedForward(X)
        X += X_f   
        X = self.norm2(X)
           
        return X


#in future may be fun to implement dropout 
import torch as tc
import MultiHead as mh
import FeedForward as FFN

def test_encoder_layer():
    h = 8  
    d_model = 64 
    ffn_dim = 256  
    seq_length = 20 
    batch_size = 10 
    
    encoder_layer = EncoderLayer(h, d_model, ffn_dim)
    X = tc.rand(batch_size, seq_length, d_model)
    expected_output_shape = (batch_size, seq_length, d_model)

    try:
        output = encoder_layer(X)

        assert output.shape == expected_output_shape, (
            f"shape {output.shape} does not match expected {expected_output_shape}"
        )

        print("Test passed: utput shape correct and no errors...I'm surprised as well ")

    except Exception as e:
        print(f"An error occurred: {e}")

#test_encoder_layer()