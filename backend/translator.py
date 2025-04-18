import torch
import math
import numpy as np
from typing import Tuple, List

# Constants
NEG_INFTY = -1e9

def create_masks(eng_batch: Tuple[str], esp_batch: Tuple[str], max_sequence_length: int) -> Tuple:
    """
    Create masking tensors for the transformer model.
    
    Args:
        eng_batch: Batch of English sentences
        esp_batch: Batch of Spanish sentences
        max_sequence_length: Maximum sequence length for padding
        
    Returns:
        Tuple of masking tensors for encoder self-attention, decoder self-attention, and decoder cross-attention
    """
    num_sentences = len(eng_batch)
    
    # Create look-ahead mask for decoder
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    
    # Initialize padding masks
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    # Fill padding masks
    for idx in range(num_sentences):
        eng_sentence_length, esp_sentence_length = len(eng_batch[idx]), len(esp_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        esp_chars_to_padding_mask = np.arange(esp_sentence_length + 1, max_sequence_length)
        
        # Set padding positions to True
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, esp_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, esp_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, esp_chars_to_padding_mask, :] = True

    # Convert padding masks to attention masks
    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

def translate_with_model(model, eng_sentence: str, max_sequence_length: int, ind_to_esp: dict, end_token: str) -> str:
    """
    Translate an English sentence to Spanish using the transformer model.
    
    Args:
        model: The transformer model
        eng_sentence: English sentence to translate
        max_sequence_length: Maximum sequence length
        ind_to_esp: Index to Spanish token mapping
        end_token: End of sentence token
        
    Returns:
        Translated Spanish sentence
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    
    # Convert to batch format (tuples)
    eng_sentence = (eng_sentence,)
    esp_sentence = ("",)
    
    # Generate translation one token at a time
    for word_counter in range(max_sequence_length):
        # Create masks
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_sentence, esp_sentence, max_sequence_length
        )
        
        # Generate predictions
        predictions = model(
            eng_sentence,
            esp_sentence,
            encoder_self_attention_mask.to(device),
            decoder_self_attention_mask.to(device),
            decoder_cross_attention_mask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=False
        )
        
        # Get next token prediction
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = ind_to_esp[next_token_index]
        
        # Add token to generated sentence
        esp_sentence = (esp_sentence[0] + next_token, )
        
        # Stop if end token is generated
        if next_token == end_token:
            break
    
    # Return the translated sentence (without special tokens)
    return esp_sentence[0]