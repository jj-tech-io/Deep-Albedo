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
### Related Videos
- [Thesis Presentation Video](https://www.youtube.com/watch?v=2eaYhO5JoIg&ab_channel=JoelJohnson)

### Read Thesis Publication
Currently in the final process of being published, date TBA

### References

1. Joel Johnson, Kenneth Chau, Wei Sen Loi, Abraham Beauferris, Swati Kanwal, and Yingqian Gu. "Deep Albedo: A Spatially Aware Autoencoder Approach to Interactive Human Skin Rendering." In SIGGRAPH Asia 2023 Posters, SA '23, New York, NY,USA, 2023. Association for Computing Machinery. [Link](https://example.com)
2. Joel Johnson, Wei Sen Loi. "Invited Workshop, W3: AI for Digital Humans at the 38th AAAI Conference on Artificial Intelligence (AAAI-24)," Vancouver, Canada, February 20-27, 2024. [Link](https://example.com)
3. C. Aliaga, C. Hery, and M. Xia. "Estimation of Spectral Biophysical Skin Properties from Captured RGB Albedo." 2022. [arXiv:2201.10695](https://arxiv.org/abs/2201.10695).
4. V. H. Aristizabal-Tique et al. "Facial Thermal and Blood Perfusion Patterns of Human Emotions: Proof-of-Concept." J. Thermal Biology, 112:103464, 2023.
5. J. Arvo and D. Kirk. "Particle Transport and Image Synthesis." Computer Graphics, 24(4):63-66, 1990.
6. A. N. Bashkatov et al. "Optical Properties of Skin, Subcutaneous, and Muscle Tissues: A Review." J. Innovative Optical Health Sciences, 2011.
7. S. Chen and W. Guo. "Auto-encoders in Deep Learning." Mathematics (Basel),2023.
8. C. Donner and H. Wann Jensen. "A Spectral BSSRDF for Shading Human Skin." Rendering Techniques, 409-418, 2006.

