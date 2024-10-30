import torch as tc
import math 
##I'm assuming the d_model has even dimentions here to make my life slightly easier, also original paper used 512 
class PositionalEncoder(tc.nn.Module):
    def __init__(self, d_model, max_length = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        position = tc.arange(0, max_length, dtype=tc.float).unsqueeze(1)
        denom = tc.exp(tc.arange(0, d_model, 2).float()/ d_model * (-math.log(10000.0))) ##
        
        pe = tc.zeros(max_length, d_model)

        pe[:, 0::2] = tc.sin(position * denom)#all,   1
        pe[:, 1::2] = tc.cos(position * denom) #all 2 

        self.register_buffer('pe', pe) #no training

#P E(pos,2i) = sin(pos/100002i/dmodel)
#P E(pos,2i+1) = cos(pos/100002i/dmodel)
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x


def test_positional_encoding():
    d_model = 512
    max_length = 20
    pe_layer = PositionalEncoder(d_model=d_model, max_length=max_length)

    input_tensor = tc.zeros((2, max_length, d_model))  
    output_tensor = pe_layer(input_tensor)
    assert output_tensor.shape == input_tensor.shape, "dimensions do not match input"
    pos = 0
    denom_even = tc.exp(tc.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    expected_pe = tc.sin(pos * denom_even)
    assert tc.allclose(output_tensor[0, pos, 0::2], expected_pe), "incorrect encoding at pos 0"
    pos = 1
    denom_odd = tc.exp(tc.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    expected_pe = tc.cos(pos * denom_odd)
    print("tests passed")

