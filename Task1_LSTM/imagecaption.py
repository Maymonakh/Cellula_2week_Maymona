from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


# Create image caption
def generate_caption(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    output = model.generate(
        **inputs,
        max_new_tokens=30
    )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption