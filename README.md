# ml-from-Scratch

# let's write me optimize pyhton code 

## TODO :
1. CNN
2. RNN
3. LSTM
4. Bi-directional
5. Seq-Seq
6. Transformer 
7. Vit
8. llama -2 
9. stabale - difussion 
10. GANs:


# Transformer - first paper - > Attention is all you need

## Introduction

This code implements a Transformer model, a popular architecture for sequence-to-sequence tasks, such as machine translation. The Transformer consists of several key components, including input embedding layers, positional embedding layers, layer normalization, feedforward layers, multi-head attention, residual connections, encoder blocks, decoder blocks, and a linear layer for projection. The model is designed to be modular and customizable.

## Code Structure

The code is organized into several classes:

1. **InputEmbeddinglayer:** This class represents the input embedding layer, which maps input sentences to vectors based on word representations.

2. **postionalembeddinglayer:** This class applies positional embedding to the input sequences to consider the order of words in a sentence.

3. **layernorm:** This class performs layer normalization on the input.

4. **feedforwardlayer:** This class implements the feedforward layer, a component of the Transformer architecture.

5. **Multiheadattention:** This class implements the multi-head attention mechanism as described in the original Transformer paper.

6. **residualconnection:** This class applies residual connections to the input.

7. **Encoderblock:** This class defines the encoder block, which consists of multi-head attention and feedforward layers.

8. **Encoder:** This class stacks multiple encoder blocks together.

9. **Decoderblock:** This class defines the decoder block, which includes self-attention, cross-attention, and feedforward layers.

10. **Decoder:** This class stacks multiple decoder blocks together.

11. **linearlayer:** This class applies a linear layer for projection.

12. **Transformer:** This class combines the encoder, decoder, and other components to create the Transformer model.

## Usage

To use this code to create a Transformer model, you can follow these steps:

1. Import the necessary modules:

```python
import torch
import torch.nn as nn
import math

enc_vocab_size = # specify the size of the encoder vocabulary
dec_vocab_size = # specify the size of the decoder vocabulary
enc_seq_len = # specify the length of the encoder sequence
dec_seq_len = # specify the length of the decoder sequence

transformer = build_trarnsformer(enc_vocab_size, dec_vocab_size, enc_seq_len, dec_seq_len)


# Import necessary modules
import torch
import torch.nn as nn
import math

# Copy and paste the provided code into your project

# Create an instance of the Transformer model
enc_vocab_size = 10000
dec_vocab_size = 10000
enc_seq_len = 50
dec_seq_len = 50

transformer = build_trarnsformer(enc_vocab_size, dec_vocab_size, enc_seq_len, dec_seq_len)

# Use the transformer for training, inference, etc.
# ...
