from transformers import CLIPProcessor, CLIPModel

def download_clip_model():
    # Download the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Save them locally
    model.save_pretrained("./local_clip_model")
    processor.save_pretrained("./local_clip_processor")

if __name__ == "__main__":
    download_clip_model()
