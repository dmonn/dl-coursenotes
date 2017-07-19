# Variational Auto Encoder Notes

## Purpose

- Reconstruct output
	- Encoder and Decoder
	- Instead of Direct Link ->  sampling layer which samples form a distribution (usually a Gaussian) and then feeds the generated samples to the decoder part.
- Can generate pictures with specific features
- GANs only choose between "fake" and "real", not realism or object
- Example Cat: http://kvfrans.com/variational-autoencoders-explained/
- Example Digits

## Implementation

http://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

## Usage

Pixel Art generation: https://mlexplained.wordpress.com/category/generative-models/vae/
Generation of 3D Models for a video game: https://arxiv.org/pdf/1606.05908.pdf