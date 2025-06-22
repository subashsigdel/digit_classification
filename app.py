import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Parameters - same as training
latent_dim = 100
num_classes = 10
image_size = 28

# Define Generator class (must match training)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(out.size(0), 1, image_size, image_size)

# Load the trained generator model
@st.cache(allow_output_mutation=True)
def load_model():
    model = Generator()
    model.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

G = load_model()

# Function to generate images for a digit
def generate_images(generator, digit, n=5):
    z = torch.randn(n, latent_dim)
    labels = torch.tensor([digit]*n)
    with torch.no_grad():
        gen_imgs = generator(z, labels).cpu()
    # Denormalize images from [-1,1] to [0,255]
    gen_imgs = (gen_imgs + 1) / 2
    gen_imgs = gen_imgs.clamp(0,1)
    pil_images = []
    for img_tensor in gen_imgs:
        img_np = img_tensor.squeeze().numpy() * 255
        img = Image.fromarray(img_np.astype(np.uint8), mode='L')
        pil_images.append(img)
    return pil_images

# Streamlit UI
st.title("Handwritten Digit Generator")

digit_input = st.text_input("Enter a digit (0-9) to generate", "0")

# Validate input
if digit_input.isdigit() and 0 <= int(digit_input) <= 9:
    digit = int(digit_input)
    if st.button("Generate 5 images"):
        images = generate_images(G, digit, n=5)
        st.write(f"Generated 5 images of digit {digit}:")
        cols = st.columns(5)
        for col, img in zip(cols, images):
            col.image(img, width=100)
else:
    st.warning("Please enter a valid digit between 0 and 9.")

