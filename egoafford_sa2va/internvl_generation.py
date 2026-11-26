"""InternVL native image-token replacement with matching embedding dtype.

Adapted from OpenGVLab InternVL3-8B modeling_internvl_chat.py, generate().
The vision encoder and language generation stay native. Only the replacement
tensor's dtype is matched to the language embeddings (FP32 after PEFT setup).
"""
import torch


@torch.inference_mode()
def generate(model, pixel_values, input_ids, attention_mask, **kwargs):
    vision = model.extract_feature(pixel_values)
    embeddings = model.language_model.get_input_embeddings()(input_ids)
    batch, length, channels = embeddings.shape
    embeddings = embeddings.reshape(batch * length, channels)
    selected = input_ids.reshape(-1) == model.img_context_token_id
    embeddings[selected] = vision.reshape(-1, channels).to(embeddings)
    return model.language_model.generate(
        inputs_embeds=embeddings.reshape(batch, length, channels),
        attention_mask=attention_mask,
        **kwargs,
    )
