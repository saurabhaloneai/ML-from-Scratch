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
    parameters : d_model : int : size of the vector,
    
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
        

        
        
        
class Encoder(nn.Module):
    '''
    This will take the input and apply the encoder
    paramaters : layers : nn.ModuleList : list of layers
    
    '''
    def __init__(self,layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layernorm(layers[0].d_model)
        
    def forward(self, x, mask_enc):
        for layer in self.layers:
            x = layer(x, mask_enc)
        return self.norm(x)
    
    
class Decoderblock(nn.Module):
    '''
    This block will take the input and apply the decoder block
    Parameters : self_attention : Multiheadattention : self attention layer
    '''
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
        
        x = self.residualconnections[0](x, lambda x: self.self_attention(x, x, x, mask_dec))
        x = self.residualconnections[1](x, lambda x: self.cross_attention(x, enc_output, enc_output, mask_enc))
        x = self.residualconnections[2](x, self.feedforward)
        return x
        

# decoder

class Decoder(nn.Module):
    '''
    this block will take the input and apply the decoder
    parameters : layers : nn.ModuleList : list of layers
    
    '''
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layernorm()
        
    def forward(self, x, enc_output, mask_enc, mask_dec):
        for layer in self.layers:
            x = layer(x, enc_output, mask_enc, mask_dec)
        return self.norm(x)
    
class linearlayer(nn.Module):
    '''
    This will take the input and apply the linear layer
    parameters : d_model : int : size of the vector, vocab_size : int : size of the vocablury
    
    '''
    
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim = -1)
    
    
#Transformer block 

class Transformer(nn.Module):
    '''
    This block will combine the encoder and decoder and submodules
    parameters : encoder : Encoder : encoder block, decoder : Decoder : decoder block,
    '''
    
    def __init__(self, encoder : Encoder, decoder : Decoder, dec_embed: InputEmbeddinglayer, enc_embed: InputEmbeddinglayer,pos_dec : postionalembeddinglayer, pos_enc: postionalembeddinglayer, linear: linearlayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dec_embed = dec_embed
        self.enc_embed = enc_embed
        self.pos_dec = pos_dec
        self.pos_enc = pos_enc
        self.linear = linear
        
    def encoder(self, src, mask_enc):
        return self.encoder(self.enc_embed(src) + self.pos_enc(src), mask_enc)
    
    def decoder(self, trg, enc_output, mask_enc, mask_dec):
        return self.decoder(self.dec_embed(trg) + self.pos_dec(trg), enc_output, mask_enc, mask_dec)
    
    def project(self, x):
        return self.linear(x)
    

def build_trarnsformer(enc_vocab_size : int, dec_vocab_size : int, enc_seq_len: int, dec_seq_len: int ,d_model: int =512, h:int =8 , N : int =6, dropout : float = 0.1, d_ff: int = 2048):
    '''
    this block will build the transformer
    parameters : enc_vocab_size : int : size of the encoder vocablury, dec_vocab_size : int : size of the decoder vocablury,
    
    '''
    # creats the embedding layer 
    
    enc_embed = InputEmbeddinglayer(d_model, enc_vocab_size)
    dec_embed = InputEmbeddinglayer(d_model, dec_vocab_size)
    
    # creates the positional embedding layer
    
    pos_enc = postionalembeddinglayer(d_model, enc_seq_len, dropout)
    pos_dec = postionalembeddinglayer(d_model, dec_seq_len, dropout)
    
    # creats the encoder and decoder block 
    
    encoder_block = []
    
    for _ in range(N):
        
        enc_atten = Multiheadattention(d_model, h, dropout)
        feed_forward = feedforwardlayer(d_model, d_ff, dropout)
        encoder = Encoderblock(enc_atten, feed_forward, dropout)
        # encoder_block.append(Encoderblock(d_model, h, d_ff, dropout))
        # feed_forward = feedforwardlayer(d_model, d_ff, dropout)
        # encoder = Encoder(nn.ModuleList(encoder_block))
        encoder.append(encoder_block)
        
    decoder_block = []
    
    for _ in range(N):
        dec_atten = Multiheadattention(d_model, h, dropout)
        dec_cross_atten = Multiheadattention(d_model, h, dropout)
        feed_forward = feedforwardlayer(d_model, d_ff, dropout)
        decoder = Decoderblock(dec_atten, dec_cross_atten, feed_forward, dropout)
        decoder.append(decoder_block)
        
    # creates the encoder and decoder
    
    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))
    
    #PREOJCTION LAYER   
    
    linear_layer = linearlayer(d_model, dec_vocab_size)
    
    # creates the transformer
    
    transformer = Transformer(encoder, decoder, dec_embed, enc_embed, pos_enc , pos_dec,linear_layer)
    
    # paramter initialization
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer