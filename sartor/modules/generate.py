import torch
from PIL import Image

def generate(model,
             processor,
             tokenizer,
             img_path,
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

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        out_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
