# stylegan

- StyleGAN
- StyleGAN2
- StyleGAN2 + ada
- FreezeG
- FreezeD
- StyleGAN3

StyleGAN MODEL : https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT

---

# 0

## Generate

```
python generate.py --sample N_FACES --pics N_PICS --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt
```

## Closed-Form Factorization

```
python closed_form_factorization.py ./checkpoint/stylegan2-ffhq-config-f.pt
```

## Apply Factor

```
python apply_factor.py -i 19 -d 5 -n 10 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt factor.pt
```

Will generate 10 random samples, and samples generated from latents that moved along 19th eigenvector with size/degree +-5.

---

# 1

```
python prepare_data.py --path ../../datasets/animal --out ./data 
```

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 train.py --path ./data
```

---

# Reference

- [https://github.com/rosinality](https://github.com/rosinality)
- [https://github.com/bryandlee/FreezeG](https://github.com/bryandlee/FreezeG)
- [https://github.com/sangwoomo/FreezeD](https://github.com/sangwoomo/FreezeD)
