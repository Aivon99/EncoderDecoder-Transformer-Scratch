import torch as tc

class FeedForward(tc.nn.Module):
    def __init__(self, ) #size is defined in the paper but will keep it generalised here, not sure why, my uni student brain can't do otherwise  
        super().__init__()


# fully connected feed-forward network, which is applied to each position separately and identically.
# consists of two linear transformations with a ReLU activation in between.

# FFN(x) = max(0, xW1 + b1)W2 + b2 (2)

# While the linear transformations are the same across different positions, they use different parameters
# from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
# The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
# df f = 2048.