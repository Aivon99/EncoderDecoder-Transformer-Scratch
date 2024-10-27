import torch as tc
import SelfBuilt as sb

class MultiHead(nn.Module):
    def __init__(self, h, d_model ): #projections d_k = d_v = d_model/h, in the paper's case 64 
        super().__init__()
 
        self.W_O = tc.nn.Linear(d_model, d_model, bias = False) # d_model x d_model
        self.W_Q = tc.nn.Linear(d_model, d_model/h, bias = False) # d_model x d_k 
        self.W_K = tc.nn.Linear(d_model, d_model/h, bias = False) # d_model x d_k
        self.W_V = tc.nn.Linear(d_model, d_model/h, bias = False) # d_model x d_v

        
    def forward(self, V, K, Q, flag = "none"):
        
        Q_p = tc.matmul(self.W_Q, Q)
        V_p = tc.matmul(self.W_V, V)
        K_p = tc.matmul(self.W_k, K)

        return tc.matmul(sb.ScaleDotProduct(Q_p, K_p, V_p, flag))

    #not needed to define get/set methods pytorch nn tracks the parameters (thank god) 
#TODO: test 