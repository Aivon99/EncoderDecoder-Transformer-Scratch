import torch as tc

def ScaleDotProduct(Q, K, V, flag):
    QK = tc.matmul(Q, tc.transpose(K)) 
    d_k = len(Q) #add equivalent of .length() function 
    QK = QK/(d_k**(1/2))
    scores = tc.softmax(QK) #TODO: use the flag var to do masking (or not) finish buir

    return tc.matmul(scores, V)

