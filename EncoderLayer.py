import torch as tc
import MultiHead as mh
import FeedForward as FFN
class EncoderLayer(tc.nn.Module):
    
    def __init__(self, h, d_model):
        super().__init__()
        self.multiHead = mh.__init__( h, d_model)



        tc.layer_norm