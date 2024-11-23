# PyTorch GANs :computer: :art:
This repo contains PyTorch implementation of Vanilla GAN architecture. <br/>

## Table of Contents
  * [Understanding GANs](#understanding-gans)
  * [Vanilla GAN](#vanilla-gan)
  * [GAN Loss Function](#gan-loss-function)

## Understanding GANs

GAN stands for Generative Adversarial Networks, which is a type of deep learning model that consists of two networks: a generator and a discriminator. The generator network learns to generate realistic-looking fake data (e.g. images, audio, text) from random noise, while the discriminator network learns to distinguish the fake data from the real data. The two networks are trained simultaneously in an adversarial manner, where the generator tries to fool the discriminator by generating more realistic fake data, while the discriminator tries to correctly identify the real and fake data.

The original paper introducing GANs is titled [Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) and was authored by Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. It was published in 2014 at the Conference on Neural Information Processing Systems (NIPS).

GANs have two components:
1. <b>Generator Network:</b> The generator network samples from an Isotropic Gaussian distribution and applies a transformation so that the resulting distribution mimics the data distribution.
2. <b>Discriminator Network:</b> The discriminator network is a classifier trained to discriminate between real and generated samples.

## Vanilla GAN

Vanilla GAN is my implementation of the original GAN paper with certain modifications mostly in the model architecture,
like the usage of LeakyReLU and 1D batch normalization.

## GAN Loss Function

### Total GAN Loss

The GAN loss is a min-max optimization problem which is why it is also known as adversarial loss. $`p_{\text{data}}(x)`$ is the data distribution and $`p_z(z)`$ is the model distribution. Like any other generative model, the goal is to minimize some kind of divergence metric between these two distributions. GAN loss can be seen as a minimization of a general class of divergence metrics called f-divergences. The final loss is given as:

```math
\mathcal{L}_{\text{GAN}} = \min_{\phi} \max_{\theta} \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D_{\theta}(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D_{\theta}(G_{\phi}(z)))]
```

### Discriminator Loss

The optimization problem for the discriminator is:

```math
\mathcal{L}_D = \max_{\theta} \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D_{\theta}(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D_{\theta}(G_{\phi}(z)))]
```

```math
\mathcal{L}_D = \min_{\theta} - \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D_{\theta}(x)] - \mathbb{E}_{z \sim p_z(z)}[\log(1 - D_{\theta}(G_{\phi}(z)))]
```

### Generator Loss

The loss for the generator is simplified as:

```math
\mathcal{L}_G = \min_{\phi} \mathbb{E}_{z \sim p_z(z)}[\log(1 - D_{\theta}(G_{\phi}(z)))]
```

```math
\mathcal{L}_G = \max_{\phi} \mathbb{E}_{z \sim p_z(z)}[\log D_{\theta}(G_{\phi}(z))]
```


```math
\mathcal{L}_G = \min_{\phi} - \mathbb{E}_{z \sim p_z(z)}[\log D_{\theta}(G_{\phi}(z))]
```

The expectations in the above equations are computed using Monte Carlo approximations by taking sample averages. 

## Sample Outputs

GAN was trained on data from the MNIST dataset. Here is how the generated digits look like:

<div style="display: flex;">
    <img src="data/generated_imagery/generated_image0.jpg" style="width: 30%; display: inline-block;">
    <img src="data/generated_imagery/generated_image.jpg" style="width: 30%; display: inline-block;">
</div>

---

## Acknowledgements

I've used the following repositories as reference for implementing my version:
* [pytorch-GANs](https://github.com/gordicaleksa/pytorch-GANs) (PyTorch)
* [research_implementations](https://github.com/ahmadchalhoub/research_implementations) (PyTorch)
