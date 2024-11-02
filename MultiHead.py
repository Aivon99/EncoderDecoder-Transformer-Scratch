import torch as tc
import SelfBuilt as sb

class MultiHead(tc.nn.Module):
        
    def __init__(self, h, d_model ): #projections d_k = d_v = d_model/h, in the paper's case 64, expecting d_model%h -> 0 
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h
         
        self.W_O = tc.nn.Linear(self.d_model, self.d_model, bias = False) # d_model x d_k 
        self.W_Q = tc.nn.Linear(self.d_model, self.d_model, bias = False) # d_model x d_k 
        self.W_K = tc.nn.Linear(self.d_model, self.d_model, bias = False) # d_model x d_k
        self.W_V = tc.nn.Linear(self.d_model, self.d_model, bias = False) # d_model x d_v
    
        
    def forward(self, V, K, Q, mask = None):
        batch_size, seq_length, _ = Q.size()  #
        
    #epected input: Q (batch_size, seq_length, d_model)
        
        
        Q_p = self.W_Q(Q).view(batch_size, seq_length, self.h, self.d_k).transpose(1, 2)
        K_p = self.W_K(K).view(batch_size, seq_length, self.h, self.d_k).transpose(1, 2)
        V_p = self.W_V(V).view(batch_size, seq_length, self.h, self.d_k).transpose(1, 2) #originally thought of doing a "single head" with mats dim D_model X d_v (as per paper) but ultimately found too unpractical

      
        scaled_attention_output = sb.ScaleDotProduct(Q_p, K_p, V_p, mask)  # Should return a tensor of shape (batch_size, h, seq_length, d_k)

         
        concat_attention = scaled_attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_k * self.h) #contigus?
        output = self.W_O(concat_attention)
        if tc.isnan(output).any():
            print("NaNs found in multiheadOutput")

        return output


    
#TODO: test 
def test_multihead_attention():
    h = 8  
    d_model = 64  
    seq_length = 20  
    multi_head_attention = MultiHead(h, d_model)
    batch_size = 10 
    Q = tc.rand(batch_size, seq_length, d_model)  #1600
    K = tc.rand(batch_size, seq_length, d_model)  
    V = tc.rand(batch_size, seq_length, d_model)  
    print(V.size())
    try:
    
        output = multi_head_attention(V, K, Q)

        expected_output_shape = (batch_size, seq_length, d_model)
        assert output.shape == expected_output_shape, f"Output shape {output.shape} does not match expected {expected_output_shape}"

        print("Test passed: Output shape is correct.")

    except Exception as e:
        print(f"An error occurred: {e}")

#test_multihead_attention()
