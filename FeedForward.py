import torch as tc

class FeedForward(tc.nn.Module):
    def __init__(self, d_model, d_2 = 2048):   #size is defined in the paper but will keep it kinda generalised here, not sure why, my uni student brain can't do otherwise  
        
        
        super().__init__()
        self.d_model = d_model
        self.d_2 = d_2
        self.W1 = tc.nn.Linear(d_model, d_2 ,bias = True)
        self.W2 = tc.nn.Linear(d_2 ,d_model, bias = True)
        self.relu = tc.nn.ReLU()
    
    
    
    def forward(self, X):
        #expected input dim (batch_size, seq_length, d_model)
        X_t1 = self.relu(self.W1(X))   #(batch_size, seq_length, d_2)
        return self.W2(X_t1)   #(batch_size, seq_length, d_model)
    

def test_feedforward():
    batch_size = 4
    seq_length = 10
    d_model = 512
    d_2 = 2048

    model = FeedForward(d_model, d_2)
    X = tc.randn(batch_size, seq_length, d_model)
    output = model(X)

    assert output.shape == (batch_size, seq_length, d_model), (
        f"Expected output shape to be {(batch_size, seq_length, d_model)}, "
        f"but got {output.shape}"
    )

    print(" dimensions are as expected.")


# fully connected feed-forward network, which is applied to each position separately and identically.
# consists of two linear transformations with a ReLU activation in between.

# FFN(x) = max(0, xW1 + b1)W2 + b2 (2)

# While the linear transformations are the same across different positions, they use different parameters
# from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
# The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
# df f = 2048.

