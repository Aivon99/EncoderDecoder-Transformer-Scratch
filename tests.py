import torch 
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
from Translator import Translator


def test_encoder_layer():
    batch_size, seq_length, d_model = 10, 20, 64
    h, ffn_dim = 8, 256

    encoder_layer = EncoderLayer(h, d_model, ffn_dim)
    input_tensor = torch.rand(batch_size, seq_length, d_model)
    output = encoder_layer(input_tensor)

    print("Encoder Layer Output Shape:", output.shape)
    assert output.shape == (batch_size, seq_length, d_model), \
        f"Expected {(batch_size, seq_length, d_model)}, got {output.shape}"
    assert not torch.isnan(output).any(), "NaN values found in Encoder Layer output."

def test_decoder_layer():
    batch_size, seq_length, d_model = 10, 20, 64
    h, ffn_dim = 8, 256

    decoder_layer = DecoderLayer(h, d_model, ffn_dim)
    input_tensor = torch.rand(batch_size, seq_length, d_model)
    encoder_output = torch.rand(batch_size, seq_length, d_model)
    output = decoder_layer(input_tensor, encoder_output)

    print("Decoder Layer Output Shape:", output.shape)
    assert output.shape == (batch_size, seq_length, d_model), \
        f"Expected {(batch_size, seq_length, d_model)}, got {output.shape}"
    assert not torch.isnan(output).any(), "NaN values found in Decoder Layer output."

def test_translator():
    vocab_size, d_model, h, ffn_dim, num_layers = 5000, 64, 8, 256, 4
    batch_size, input_seq_length, output_seq_length = 10, 20, 20

    translator = Translator(vocab_size, d_model, h, ffn_dim, num_layers)
    input_tensor = torch.rand(batch_size, input_seq_length, d_model)
    output_tensor = torch.rand(batch_size, output_seq_length, d_model)
    result = translator(input_tensor, output_tensor)

    print("Translator Model Output Shape:", result.shape)
    assert result.shape == (batch_size, output_seq_length, vocab_size), \
        f"Expected {(batch_size, output_seq_length, vocab_size)}, got {result.shape}"
    assert not torch.isnan(result).any(), "NaN values found in Translator Model output."

print("Testing Encoder Layer...")
test_encoder_layer()

print("\nTesting Decoder Layer...")
test_decoder_layer()

print("\nTesting Translator Model...")
test_translator()
