import torch as tc
import Encoder
import Decoder

class Translator(tc.nn.Module):
    def __init__(self, h, d_model, ffn_dim, vocab_size):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.vocab_size = vocab_size
        self.encoder = Encoder.Encoder(h, d_model, ffn_dim, vocab_size)
        self.decoder = Decoder.Decoder(h, d_model, ffn_dim, vocab_size)
        
        self.final_projection = tc.nn.Linear(d_model, vocab_size)

    def forward(self, Inputs, Outputs):
        encoder_output = self.encoder(Inputs)  
        product = self.decoder(Outputs, encoder_output)  

        product = self.final_projection(product)  # Expected shape: (batch_size, seq_length, vocab_size)
        
        print("Shape before softmax:", product.shape)
        
        sum_check = product.max(dim=-1)
        print("Sum along vocab dimension (should be close to 1):", sum_check)
        product = tc.softmax(product, dim=-1)
        

        return product
    def check_initialization(self):
            for name, param in self.named_parameters():
                if param.requires_grad and tc.isnan(param).any():
                    print(f"NaNs found in parameter: {name}")
                else:
                    print(f"Parameter {name} initialized correctly.")

  

 
def test_translator():
    h = 8
    d_model = 64
    ffn_dim = 256
    vocab_size = 5000
    seq_length = 20
    batch_size = 10

    translator = Translator(h, d_model, ffn_dim, vocab_size)
    translator.check_initialization()
    Inputs = tc.randint(0, vocab_size, (batch_size, seq_length))
    Outputs = tc.randint(0, vocab_size, (batch_size, seq_length))

    output = translator.forward(Inputs, Outputs)

    assert output.shape == (batch_size, seq_length, vocab_size), \
    f"Expected {(batch_size, seq_length, vocab_size)}, got {output.shape}"
    assert tc.allclose(output.sum(dim=-1), tc.ones_like(output.sum(dim=-1))), "Output after softmax is not a valid probability distribution"
    print("Translator test passed")

test_translator()
 