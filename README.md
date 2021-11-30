# stylegan code collection

- 0 : StyleGAN2
- 1 : StyleGAN2 finetune (freeze G, freeze D)
- 2 : StyleClip
- 3 : StyleSpace
- 4 : PSP
- 5 : E4E

---

**StyleGAN MODEL** : https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT

## Generate

```
python 0_0_generate.py --sample N_FACES --pics N_PICS --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt
```

## Closed-Form Factorization

```
python 0_1_closed_form_factorization.py --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --out factor.pt
```

## Apply Factor

```
python 0_2_apply_factor.py -i 19 -d 5 -n 10 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --factor factor.pt
```

Will generate 10 random samples, and samples generated from latents that moved along 19th eigenvector with size/degree +-5.

## Projector

```
python 0_3_projector.py ./results/generate.png
```
---

## prepare data

```
python 1_0_prepare_data.py --path [YOUR DATASET] --out ./data 
python 1_0_prepare_data.py --path ../../datasets/animal --out ./data 
```

---

## freeze G

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 1_1_train_freeze_G.py --path ./data --ckpt ./checkpoint/stylegan2-ffhq.pt 
```

## freeze D

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 1_2_train_freeze_D.py --freezeD --path ./data --ckpt ./checkpoint/stylegan2-ffhq.pt 
```

---

## StyleCLIP

**StyleCLIP MODEL** : [https://github.com/orpatashnik/StyleCLIP](https://github.com/orpatashnik/StyleCLIP)

```
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
```

```
python 2_1_styleclip.py
```

---

## StyleSpace

**StyleSpace MODEL** : [https://github.com/xrenaa/StyleSpace-pytorch](https://github.com/xrenaa/StyleSpace-pytorch)

```
python 3_stylespace.py
```

---

## Pixel2Style2Pixel

**PSP MODEL** : [https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

```
python 4_psp.py
```

---

## Encoding4Editing

**E4E MODEL** : [https://github.com/omertov/encoder4editing](https://github.com/omertov/encoder4editing)

```
python 5_e4e.py
```

---

# Reference

- [https://github.com/rosinality](https://github.com/rosinality) - main
- [https://github.com/bryandlee/FreezeG](https://github.com/bryandlee/FreezeG) - freezeG
- [https://github.com/sangwoomo/FreezeD](https://github.com/sangwoomo/FreezeD) - freezeD
- [https://github.com/orpatashnik/StyleCLIP](https://github.com/orpatashnik/StyleCLIP) - styleclip
- [https://github.com/xrenaa/StyleSpace-pytorch](https://github.com/xrenaa/StyleSpace-pytorch) - stylespace
- [https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) - psp
- [https://github.com/omertov/encoder4editing](https://github.com/omertov/encoder4editing) - e4e

