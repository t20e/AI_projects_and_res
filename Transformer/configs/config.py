class TransformerConfig:
    # TODO Add Docstring
    d_model = 512 # All sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512
    h_heads = 8
    dropout = 0.1

    # Encoder & Decoder
    enc_dec_N = 6 # num of stacks of encoders and decoders

