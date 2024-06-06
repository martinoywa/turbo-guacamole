import io
from torchvision import transforms
from PIL import Image

# hyperparameters
n = 768 // 2
mean, std = [0.6703, 0.5346, 0.8518], [0.1278, 0.1731, 0.0729]


def preprocess(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((n, n)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # image_bytes are from the uploaded image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # sends a single image batch
    return transform(image).unsqueeze(0)
