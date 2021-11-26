import os
import torch
import torchvision.transforms as transforms
import torchvision
import pprint
import src.augmentations as augmentations
from PIL import Image
from argparse import Namespace
from src.models.psp import pSp

EXPERIMENT_DATA_ARGS = {
    "cars_encode": {
        "model_path": "../checkpoint/e4e_cars_encode.pt",
        "transform": transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "ffhq_encode": {
        "model_path": "../checkpoint/e4e_ffhq_encode.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS['ffhq_encode']
model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']
opts['encoder_name'] = 'e4e'
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

transformer = EXPERIMENT_ARGS['transform']

with torch.no_grad():
    tensor_image = transformer(Image.open('./results/00020.jpg'))
    tensor_image = tensor_image.unsqueeze(0).to('cuda').float()
    output, _ = net(tensor_image, randomize_noise=False, return_latents=True)
    
torchvision.utils.save_image(output.detach().cpu(), 
                             os.path.join("./results/e4e.png"), 
                             normalize=True, 
                             scale_each=True, 
                             range=(-1, 1))