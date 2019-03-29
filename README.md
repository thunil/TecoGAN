# TecoGAN
This repo will contain source code and materials for the TecoGAN project, i.e. code for a TEmporally COherent GAN.
_Authors: Mengyu Chu, You Xie, Laura Leal-Taixe, Nils Thuerey. Technical University of Munich._

We now publish the code for the _inference_ mode.
The _training_ code is still under preparation.
For now, enjoy trying our model on low resolution videos!
You can also watch the video and read pre-print of our paper!

Video: <https://www.youtube.com/watch?v=pZXFXtfd-Ak>
Preprint: <https://arxiv.org/pdf/1811.09393.pdf>

![TecoGAN teaser image](resources/teaser.jpg)

### Run
Take a look of the runGan.py file. There are explanations for parameters.

```bash
# Install tensorflow1.8+,
pip3 install --ignore-installed --upgrade tensorflow-gpu # or tensorflow
# Install PyTorch (only necessary for the metric evaluations) and other things...
pip3 install -r requirements.txt

# Download our TecoGAN model, the _Vid4_ and _TOS_ scenes showed in our paper and video.
python3 runGan.py 0

# Run the inference mode on the calendar scene
# You can take a look of the parameter explanations in the runGan.py, feel free to try other scenes!
python3 runGan.py 1 

# Evaluate the results with 4 metrics, PSNR, LPIPS[1], and our temporal metrics tOF and tLP
# Take a look of the paper for more details! 
python3 runGan.py 2

# Train TecoGAN on Video data
python3 runGan.py 3 # Coming Soon!


```
### Acknowledgements
This work was funded by the ERC Starting Grant realFlow (ERC StG-2015-637014).  
Part of the code is based on LPIPS[1] and Photo-Realistic SISR[2].

### Reference
[1] [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)](https://github.com/richzhang/PerceptualSimilarity)  
[2] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/brade31919/SRGAN-tensorflow.git)  

TUM I15 <https://ge.in.tum.de/> , TUM <https://www.tum.de/>

