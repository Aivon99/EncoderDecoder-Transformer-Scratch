import torch as tc

def ScaleDotProduct(Q, K, V, flag = "none"):
    if flag != "none":
        QK = tc.matmul(Q, tc.transpose(K)) 
        d_k = len(Q) #add equivalent of .length() function 
        QK = QK/(d_k**(1/2))
        scores = tc.softmax(QK) #TODO: implement masking, padding and cheating

        return tc.matmul(scores, V)
    else 
        return 0
