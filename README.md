# Privacy-preserving Image Recognition (and A Strong Technical Baseline)
<!-- 
This repository is an implementation of the privacy-preserving algorithm discussed in
```
Privacy-preserving Collaborative Learning with Automatic Transformation Search
```
which can be found at https://arxiv.org/abs/2011.12505
<p align="center">
    <img src="assets/framework.png" width="80%" />
</p> -->


### Abstract
Given that Machine-Learning (ML) is increasingly being used in a variety of privacy-sensitive applic- ations, the knowledge gap between ML systems and privacy issues must be bridged. To solve this problem, a number of image privacy-preserving methods have been proposed to prevent information leakage in computer vision community. In general, both the inference and privacy protection ability are evaluated to demonstrate the performance of a privacy-preserving model, with the former meas- ured by classification accuracy and the latter quantified by image quality metrics. Specifically, privacy protection ability is measured by MSE, PSNR, or SSIM, etc., , image quality metrics which measure the similarity or distance between the original and reconstructed (based on leaked information) im- ages. A method with a lower similarity (large distance) value is considered to better preserve image information. However, we find that current image quality metrics are less indicative of the quality of privacy-preserving images when the values of PSNR etc., are similar (e.g., ±1 − 2) across different methods. In other words, increasing the PSNR value of a method by one or two points may not prove that it is effective. To confirm this finding, a lot of experiments are conduced in this paper, and we introduce Human-in-the-Loop (HitL) and Model-Based Recognition (MR) as evaluation metric for privacy protection ability evaluating. On the other hand, we propose a Strong Technical Baseline, which increases classification accuracy and maintains privacy preservation.

## Code
The search algorithm file that contains  can be found at ```searchalg/```. The other important experimental part can be found at ```benchmark/```.

## Setup
You can use [anaconda](https://www.anaconda.com/distribution/) to install our setup by running
```
conda env create -f environment.yml
conda activate ats
```
