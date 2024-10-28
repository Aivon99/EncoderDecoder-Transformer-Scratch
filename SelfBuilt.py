import torch as tc
#Q,K,V : 1 x d_model (or d_k)

def ScaleDotProduct(Q, K, V, flag = "none"):
    if flag == "none":
        QK = tc.matmul(Q, K.transpose(-2, -1)) 
        d_k = Q.size(-1)    
        QK = QK/(d_k**(1/2))
        scores = tc.softmax(QK, dim=-1) # TODO: implement masking, padding and cheating

        return tc.matmul(scores, V)
    else: 
        return 0
