import torch as tc
import DecoderLayer as dl

#Not gonna put the embedding for now

class Decoder(tc.nn.Module):
    def __init__(self,  h, d_model, ffn_dim): #TODO: generalise to N layers, for now hardcoded to 6
        super().__init__()
        self.layer1 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer2 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer3 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer4 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer5 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer6 =  dl.DecoderLayer(h, d_model, ffn_dim)

        
    def forward(self, X, encoder_output):
        X_1 = self.layer1(X)
        X_2 = self.layer1(X_1, encoder_output)
        X_3 = self.layer1(X_2, encoder_output)
        X_4 = self.layer1(X_3, encoder_output)
        X_5 = self.layer1(X_4, encoder_output)
        X_6 = self.layer1(X_5, encoder_output)

        return X_6

import torch as tc

batch_size = 2       
seq_length = 10      
d_model = 64         
h = 8                
ffn_dim = 256        

X = tc.rand(batch_size, seq_length, d_model)

encoder = Encoder(h=h, d_model=d_model, ffn_dim=ffn_dim)

output = encoder(X)

assert output.shape == X.shape, f"put shape {output.shape} not match input shape {X.shape}"
assert not tc.isnan(output).any(), "output contains NaN values"
print("encoder forwar shape:", output.shape)
print("encoder forward successful, output shape matches input shape...again surprised...")