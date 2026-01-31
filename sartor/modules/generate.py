import torch
from PIL import Image

def generate(model, 
             processor, 
             tokenizer, 
             img_path,
             prompt,
             device, 
             max_new_tokens=60,
             min_new_tokens=20, 
             num_beams=4, 
             length_penalty=1.5, 
             repetition_penalty=1.2,
             ):
    
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    decoder_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = decoder_input_ids.shape[1]
    
    with torch.no_grad():
        out_ids = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()