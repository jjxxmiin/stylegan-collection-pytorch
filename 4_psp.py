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
    "ffhq_encode": {
        "model_path": "../checkpoint/psp_ffhq_encode.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "ffhq_frontalize": {
        "model_path": "../checkpoint/psp_ffhq_frontalization.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celebs_sketch_to_face": {
        "model_path": "../checkpoint/psp_celebs_sketch_to_face.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    },
    "celebs_seg_to_face": {
        "model_path": "../checkpoint/psp_celebs_seg_to_face.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.ToOneHot(n_classes=19),
            transforms.ToTensor()])
    },
    "celebs_super_resolution": {
        "model_path": "../checkpoint/psp_celebs_super_resolution.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[16]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "toonify": {
        "model_path": "../checkpoint/psp_ffhq_toonify.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS['ffhq_encode']
model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']
opts['encoder_name'] = 'psp'
pprint.pprint(opts)

opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False
if 'stylegan_size' not in opts:
    opts['stylegan_size'] = 1024

opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

transformer = EXPERIMENT_ARGS['transform']

with torch.no_grad():
    tensor_image = transformer(Image.open('./results/00020.jpg'))
    tensor_image = tensor_image.unsqueeze(0).to('cuda').float()
    output = net(tensor_image, randomize_noise=False)

torchvision.utils.save_image(output.detach().cpu(), 
                             os.path.join("./results/psp.png"), 
                             normalize=True, 
                             scale_each=True, 
                             range=(-1, 1))
    
    