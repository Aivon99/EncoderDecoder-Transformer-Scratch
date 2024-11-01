import Encoder 
import Decoder 

class Translator():
    def __init__(self, h, d_model, ffn_dim, vocab_size):
        super().__init__()

        self.h = h
        self.d_model = d_model 
        self.ffn_dim = ffn_dim
        self.vocab_size = vocab_size
        #(self,  h, d_model, ffn_dim, vocab_size
        self.encoder = Encoder.Encoder(h, d_model, ffn_dim, vocab_size)
        self.decoder = Decoder.Decoder(h, d_model, ffn_dim, vocab_size)



    def forward(self, Inputs, Outputs):
        encoder_output = self.encoder(Inputs)
        decoder_output = self.decoder(Outputs, encoder_output)
        
        return None #stud