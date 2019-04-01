# TecoGAN
This repo will contain source code and materials for the TecoGAN project, i.e. code for a TEmporally COherent GAN.
_Authors: Mengyu Chu, You Xie, Laura Leal-Taixe, Nils Thuerey. Technical University of Munich._

This repository so far contains the code for the TecoGAN _inference_ mode.
The _training_ code is still under preparation, and will follow soon.
For now, enjoy running our pre-trained model on low resolution videos! 
You can find links for downloading and instructions below.
The video and pre-print of our paper can be found here:

Video: <https://www.youtube.com/watch?v=pZXFXtfd-Ak>
Preprint: <https://arxiv.org/pdf/1811.09393.pdf>

![TecoGAN teaser image](resources/teaser.jpg)

### Additional Generated Outputs

Below you can find three additional sequences generated with TecoGAN. Our method generates fine details that 
persist over the course of long generated video sequences. E.g., the mesh structures of the armor,
the scale patterns of the lizard, and the dots on the back of the spider highlight the capabilities of our method.
Our temporal discriminator plays a key role to guide the generator network towards producing coherent detail.

<img src="resources/tecoGAN-lizard.gif" alt="Lizard" width="900"/><br>

<img src="resources/tecoGAN-armour.gif" alt="Armor" width="900"/><br>

<img src="resources/tecoGAN-spider.gif" alt="Spider" width="600" hspace="150"/><br>

### Running the TecoGAN Model

Below you can find a quick start guide.
For further explanations of the parameters take a look at the runGan.py file. 

```bash
# Install tensorflow1.8+,
pip3 install --ignore-installed --upgrade tensorflow-gpu # or tensorflow
# Install PyTorch (only necessary for the metric evaluations) and other things...
pip3 install -r requirements.txt

# Download our TecoGAN model, the _Vid4_ and _TOS_ scenes shown in our paper and video.
python3 runGan.py 0

# Run the inference mode on the calendar scene.
# You can take a look of the parameter explanations in the runGan.py, feel free to try other scenes!
python3 runGan.py 1 

# Evaluate the results with 4 metrics, PSNR, LPIPS[1], and our temporal metrics tOF and tLP with pytorch.
# Take a look of the paper for more details! 
python3 runGan.py 2

```

### Acknowledgements
This work was funded by the ERC Starting Grant realFlow (ERC StG-2015-637014).  
Part of the code is based on LPIPS[1] and Photo-Realistic SISR[2].

### Reference
[1] [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)](https://github.com/richzhang/PerceptualSimilarity)  
[2] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/brade31919/SRGAN-tensorflow.git)  

TUM I15 <https://ge.in.tum.de/> , TUM <https://www.tum.de/>

