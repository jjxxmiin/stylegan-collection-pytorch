# stylegan

- StyleGAN
- StyleGAN2
- StyleGAN2 + ada
- FreezeG
- FreezeD
- StyleGAN3

# Closed-Form Factorization

```
python closed_form_factorization.py ./checkpoint/stylegan2-ffhq-config-f.pt
```

```
python apply_factor.py -i 19 -d 5 -n 10 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt factor.pt
```

Will generate 10 random samples, and samples generated from latents that moved along 19th eigenvector with size/degree +-5.

## ref

StyleGAN MODEL : https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT