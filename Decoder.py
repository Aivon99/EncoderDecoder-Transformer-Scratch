import torch as tc
import DecoderLayer as dl
import PositionalEncoder as pe
#Not gonna put the embedding for now

class Decoder(tc.nn.Module):
    def __init__(self,  h, d_model, ffn_dim, vocab_size): #TODO: generalise to N layers, for now hardcoded to 6
        super().__init__()
        
        self.embedding = tc.nn.Embedding(vocab_size, 6)
        self.PosEncod = pe.PositionalEncoder(d_model)

        self.layer1 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer2 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer3 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer4 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer5 =  dl.DecoderLayer(h, d_model, ffn_dim)
        self.layer6 =  dl.DecoderLayer(h, d_model, ffn_dim)

        
    def forward(self, X, encoder_output):
        X = self.embedding(X)
        X = self.PosEncod(X)

        X_1 = self.layer1(X)
        X_2 = self.layer1(X_1, encoder_output)
        X_3 = self.layer1(X_2, encoder_output)
        X_4 = self.layer1(X_3, encoder_output)
        X_5 = self.layer1(X_4, encoder_output)
        X_6 = self.layer1(X_5, encoder_output)

        return X_6
