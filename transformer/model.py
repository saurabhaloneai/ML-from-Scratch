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
    
    pass