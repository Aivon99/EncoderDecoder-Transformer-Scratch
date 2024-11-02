import torch as tc
import DecoderLayer as dl
import PositionalEncoder as pe
#Not gonna put the embedding for now

class Decoder(tc.nn.Module):
    def __init__(self,  h, d_model, ffn_dim, vocab_size): #TODO: generalise to N layers, for now hardcoded to 6
        super().__init__()
        
        self.embedding = tc.nn.Embedding(vocab_size, d_model)
        self.PosEncod = pe.PositionalEncoder(d_model)

        self.layer1 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer2 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer3 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer4 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer5 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer6 =  dl.DecoderLayer(h, d_model, ffn_dim)

        
    def forward(self, X, encoder_output):
        X = X.long()
        X = self.embedding(X)
        X = self.PosEncod(X)

        X_1 = self.layer1(X, encoder_output)
        X_2 = self.layer2(X_1, encoder_output)
        X_3 = self.layer3(X_2, encoder_output)
        X_4 = self.layer4(X_3, encoder_output)
        X_5 = self.layer5(X_4, encoder_output)
        X_6 = self.layer6(X_5, encoder_output)

        return X_6

#def test_decoder():
#    h = 8 
#    d_model = 64  
#    ffn_dim = 256  
#    vocab_size = 5000  
#    seq_length = 20  
#    batch_size = 10  

#    decoder = Decoder(h, d_model, ffn_dim, vocab_size)
#    X = tc.randint(0, vocab_size, (batch_size, seq_length))
#    encoder_output = tc.rand(batch_size, seq_length, d_model)

#    output = decoder(X, encoder_output)

#    assert output.shape == (batch_size, seq_length, d_model), f"Output shape {output.shape} does not match expected shape"
#    assert not tc.isnan(output).any(), "contains NaN values"
#    print("Decoder test passed")

#test_decoder()
