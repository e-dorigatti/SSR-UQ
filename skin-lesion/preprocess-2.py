import torch
import os
from tqdm import tqdm
import torchvision


images = []
files = []
tx = torchvision.transforms.Resize((128, 128))
for f in tqdm(os.listdir("train")):
    img = torchvision.io.read_image("train/" + f)
    images.append(tx(img.float() / 255))
    files.append(f)


torch.save({
    'names': files,
    'images': torch.stack(images),
}, 'data/x_train_resized_normalized.pt')
