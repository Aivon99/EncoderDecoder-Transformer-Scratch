import torch as tc
import EncoderLayer as el
import PositionalEncoder as pe 


class Encoder(tc.nn.Module):
    def __init__(self, n_encoders,  h, d_model, ffn_dim, vocab_size = 1000): #TODO: generalise to N layers, for now hardcoded to 6
        super().__init__()
        self.n_encoders = n_encoders 
        
        self.embedding = tc.nn.Embedding(vocab_size, d_model)
        self.PosEncod = pe.PositionalEncoder(d_model)
        self.layers = tc.nn.ModuleList()
        
        for i in n_encoders:
            self.layers.append(el.EncoderLayer(h, d_model, ffn_dim))
        
    def forward(self, X):
        X = self.embedding(X.long())
        X = self.PosEncod(X)
        
        for layer in self.layers:
            X = layer(X)
        return X

 
#def encoderTester():
#    batch_size = 2       
#    seq_length = 10      
#    d_model = 64         
#    h = 8                
#    ffn_dim = 256        

#    X = tc.rand(batch_size, seq_length, d_model)
 
#    encoder = Encoder(h=h, d_model=d_model, ffn_dim=ffn_dim)

#    output = encoder(X)
#    print()
#    assert output.shape == X.shape, f"put shape {output.shape} not match input shape {X.shape}"
#    assert not tc.isnan(output).any(), "output contains NaN values"
#    print("encoder forwar shape:", output.shape)
#    print("encoder forward successful, output shape matches input shape...again surprised...")
#encoderTester()