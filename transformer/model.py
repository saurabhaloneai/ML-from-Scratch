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
    '''
    This will take the input and apply the positional embedding
    parameters : d_model : int : size of the vector, 
    seq_len : int : length of the sequence, dropout : float : dropout rate
    '''
    
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
    '''
    This will take the input and apply the layer normalization
    parameters : d_model : int : size of the vector, eps : float : epsilon
    '''
    
    
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
    '''
    This will take the input and apply the feed forward layer
    parameters : d_model : int : size of the vector,
    d_ff : int : size of the feed forward layer, dropout : float : dropout rate
    '''
    
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))
    


# multi-head attention (from paper)

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



class Multiheadattention(nn.Module):
    '''
    This will take the input and apply the multihead attention
    parameters : d_model : int : size of the vector, 
    num_heads : int : number of heads, dropout : float : dropout rate

    '''
    
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
        
    @staticmethod
    def attention(q,k,v,mask_dec,dropout: nn.Dropout):
        '''
        This will take the query, key, value and mask and apply the attention
        parameters : q : tensor : query, k : tensor : key,
        v : tensor : value, mask_dec : tensor : mask, 
        dropout : nn.Dropout : dropout layer
        '''
        
        
        d_k = q.shpae[-1]
        
        # (batch_size, num_heads, seq_len, dk)  --> (batch_size, num_heads, seq_len, seq_len)
        
        atten_scores = torch.matmul(q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask_dec is not None :
            atten_scores = atten_scores.masked_fill(mask_dec == 0, float('-1e20'))
        
        atten_scores = torch.softmax(atten_scores, dim = -1)
        
        if dropout is not None:
            atten_scores = dropout(atten_scores)
            
        return (atten_scores @ v), atten_scores

    # batch_size = q.size(0)
        
        # Q = self.WQ(q).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        # K = self.WK(k).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        # V = self.WV(v).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        
        # #attention(Q,K,V) = softmax(Q.k/ √dmodel)* V
        # # (batch_size, num_heads, seq_len, dk)
        # attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        
        # if mask_dec is not None:
        #     attention = attention.masked_fill(mask_dec == 0, float('-1e20'))
            
        # attention = torch.softmax(attention, dim = -1)
        # attention = self.dropout(attention)
        
        # # (batch_size, num_heads, seq_len, dk)
        # output = torch.matmul(attention, V)
        
        # # (batch_size, seq_len, num_heads, dk)
        # output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # return self.w0(output)
            
    
    def forward(self,q,k,v,mask_dec):
        
        query = self.w_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, num_heads, dk)
        
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.dk).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.dk).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.dk).transpose(1, 2)
        
        x, self.atten_scores = self.attention(query, key, value, mask_dec, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.dk)
        
        return self.w0(x)
   
#------------Basic buildig block ends here----------------#



# residual connection

class residualconnection(nn.Module):
    '''
    
    This will take the input and apply the residual connection
    
    '''
    
    def __init__(self, size:int, dropout:float):
        super().__init__()  
        self.norm = layernorm(size)
        self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
        
        
class Encoderblock(nn.Module):
    '''
    This will take the input and apply the encoder block
    
    '''
    
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout:float):
        super().__init__()
        
        self.Multiheadattention = Multiheadattention(d_model, num_heads, dropout)
        self.feedforwardlayer = feedforwardlayer(d_model, d_ff, dropout)
        
        self.residualconnection1 = residualconnection(d_model, dropout)
        self.residualconnection2 = residualconnection(d_model, dropout)
        
    def forward(self, x, mask_enc):
        x = self.residualconnection1(x, lambda x: self.Multiheadattention(x, x, x, mask_enc))
        x = self.residualconnection2(x, self.feedforwardlayer)
        return x
        

        
        
        
# class Encoder(nn.Module):
#     '''
#     This will take the input and apply the encoder
    
#     '''
#     def __init__(self,layers: nn.ModuleList):
#         super().__init__()
#         self.layers = layers
#         self.norm = layernorm(layers[0].d_model)
        
#     def forward(self, x, mask_enc):
#         for layer in self.layers:
#             x = layer(x, mask_enc)
#         return self.norm(x)
    
    
class Decoderblock(nn.Module):
    
    def __init__(self, self_attention: Multiheadattention, cross_attention: Multiheadattention, feedforward: feedforwardlayer, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        
        self.residualconnections = nn.Module([residualconnection(dropout)for _ in range(3)])
        
        #  self.residualconnection1 = residualconnection(self_attention.d_model, dropout)
        #  self.residualconnection2 = residualconnection(cross_attention.d_model, dropout)
        #  self.residualconnection3 = residualconnection(feedforward.d_model, dropout)
        
    def forward(self, x, enc_output, mask_enc, mask_dec):
        pass 