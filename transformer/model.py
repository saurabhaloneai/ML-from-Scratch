#input embedding layer
import torch
import torch.nn as nn 
import math

class InputEmbeddinglayer(nn.Module):
    '''
    This will just map 
    the input setentence into a vector of 
    ex.512 with represention of word in vocablury.  
    atrributes : d_model : int : size of the vector, vocab_size : int : size of the vocablury  
    '''
    
    def __init__(self, d_model:int, vocab_size:int):
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)  


#positional embedding layer

class postionalembeddinglayer(nn.Module):
    
    def __init__(self, d_model:int, seq_len:int, dropout: float):
        
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #matrix for (d_model, seq_len)  
        pe = torch.zeros(d_model, seq_len)

        #converting the position into a tesnor 
        # PE(pos,2i) =sin(pos/100002i/dmodel) -> from paper
        # PE(pos,2i+1) =cos(pos/100002i/dmodel) i is the dimension

        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # sin to even position and cos to odd position  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
       
        # (1,d_model, seq_len)
        pe  = pe.unsqueeze(0) 
        
        # pe = position embedding
        self.register_buffer('pe', pe)  # register_buffer TO STORE THE INFO ABOUT 
                                        # POSTIONAL EMEDDING IG it will be use at output 
                
                
    def forward(self, x):
        
        x = x + (self.pe[:, :x.size(1)]).requires_grad_(False)
        return self.dropout(x)
    


# layer normailzation 

class layernorm(nn.Module):
    
    
    def __init__(self, eps:float = 1e-6):
        
        super().__init__()
        
        self.eps = eps
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, x): 
        mean = x.mean(-1, keepdim = True) # u 
        std = x.std(-1, keepdim = True)
        return self.w * (x - mean) / (std + self.eps) + self.b
    
    
# feed forward layer 
# FFN(x) = max(0, xW1 + b1 )W2 + b2   

class feedforwardlayer(nn.Module):
    
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))
    


# multi-head attention 

# Attention(Q, K, V ) = softmax(Q.k/ √dmodel)* V


# MultiHead(Q, K, V ) = Concat(head1, ..., headh)W O 

# where headi = Attention(QWiQ , K WiK , V WiV )

# Where the projections are parameter matrices WiQ ∈ R dmodel ×dk , WiK ∈ R dmodel ×dk , 

# WiV ∈ Rdmodel ×dv andWO ∈Rhdv×dmodel.

# In this work we employ h = 8 parallel attention layers, or heads. 
# For each of these we use dk = dv = dmodel/h = 64. 
# Due to the reduced dimension of each head,
# the total computational cost is similar to 
# that of single-head attention with full dimensionality.



class multiheadattention(nn.Module):
    
    def __init__(self, d_model:int, num_heads:int, dropout:float):
        
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0 , "d_model should be divisible by num_heads"
        
        self.dk = d_model // num_heads
        
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        
        self.w0 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,q,k,v,mask_dec):
        
        batch_size = q.size(0)
        
        Q = self.WQ(q).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        K = self.WK(k).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        V = self.WV(v).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        
        #attention(Q,K,V) = softmax(Q.k/ √dmodel)* V
        # (batch_size, num_heads, seq_len, dk)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        
        if mask_dec is not None:
            attention = attention.masked_fill(mask_dec == 0, float('-1e20'))
            
        attention = torch.softmax(attention, dim = -1)
        attention = self.dropout(attention)
        
        # (batch_size, num_heads, seq_len, dk)
        output = torch.matmul(attention, V)
        
        # (batch_size, seq_len, num_heads, dk)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w0(output)
        
        