### Deep Albedo
# Biophysically Based Skin Color Modeling
![images/Representative Image.jpg](https://github.com/jj-tech-io/Deep-Albedo/blob/master/images/Representative%20Image.jpg)

## Introduction

This project focuses on the simulation and modeling of human skin color changes as influenced by biological and emotional factors. By utilizing Monte Carlo photon simulations and neural autoencoders, we create biophysically accurate representations of skin color changes due to aging and emotion.

The origin of skin color in this model is determined by the absorption and scattering of light, modulated by parameters such as the concentration of melanin and hemoglobin, and skin layer thickness. Our methodology involves a neural autoencoder to translate skin color into a biophysical latent space, enabling real-time adjustments that are reflected back in the sRGB color space.

## Features
- **Monte Carlo Photon Simulations**: For accurate skin spectral reflectance values.
- **Autoencoder**: Efficient real-time mapping between skin color and its biophysical properties.
- **Aging and Emotion Effects**: Modeling the impact of aging and emotional states on skin color.
- **Spatially Aware Transformations**: Learned from example images for detailed and accurate skin textures.
[Watch the Deep Albedo Introduction Video](https://clipchamp.com/watch/W12SR9O47z4)
## [To try a demo app for aging textures](https://github.com/jj-tech-io/Modify_Texture_Docker.git) <br>
[Watch the Age Texture Demo App Video](https://clipchamp.com/watch/W12SR9O47z4)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Pathlib

### Installation

1. Clone the repository:
   ```shell
   git clone https://github.com/jj-tech-io/Deep-Albedo.git
