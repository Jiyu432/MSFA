# MSFA-RSISR: Multi-Scale and Fourier Attention for Remote Sensing Image Super-Resolution

This job opportunity arises from the paper "MSFA-RSISR: Multi-Scale and Fourier Attention for Remote Sensing Image Super-Resolution PRCV2025  [[Paper](https://link.springer.com/chapter/10.1007/978-981-95-5682-3_38)].and mostly built on Recursive generalization transformer for image
super-resolution[(RGT)](https://github.com/zhengchen1999/RGT).

Given my limited coding proficiency, I am submitting this as a record. It should be noted that this code cannot be used for professional learning or work



> **Abstract:** Although Vision Transformers based on window mechanisms have shown remarkable performance in remote sensing image super resolution (RSISR), the traditional window mechanism limits the inter action of global information. Moreover, the large range and rich high frequency information of remote sensing images make reconstruction more challenging. To address these issues, this paper proposes a new RSISR framework called MSFA-RSISR based on the SwinTransformer: It innovatively introduces Frequency Fourier Block (FFB) to enhance high frequency feature extraction; designs a Multi-scale Fusion Block (MSFB) to optimize the balance between global and local features through multi scale recursion and feature compression; and employs a three-stage training strategy from natural images to remote sensing images to efficiently integrate cross-domain knowledge. Experiments on multiple datasets demonstrate that this method significantly improves the reconstruction of high frequency details and the recovery of global structures.


## Environment
- Python 3.9
- PyTorch 2.0.1

## Pretrained Models
We provide the pre-trained model for HAMD-RSISR. You can download it from the following link:
- **Baidu Drive **: [link](https://pan.baidu.com/s/1-2hgSs5knN30SahyyzeEHg?pwd=1544) 
Download the `.pth` file and place it in your designated model directory...
## database
The dataset can be downloaded from here 
- **Baidu Drive **: [link](https://pan.baidu.com/s/17kDP6AGSBh6SL8EasCgtpA?pwd=1544) 

The dataset is sourced from Uc Merced:
Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.
Shawn D. Newsam
Assistant Professor and Founding Faculty
Electrical Engineering & Computer Science
University of California, Merced
Email: snewsam@ucmerced.edu
Web: http://faculty.ucmerced.edu/snewsam/
## Citation

```
 """@inproceedings{xie2025msfa,
  title={MSFA-RSISR: Multi-scale and Fourier Attention for Remote Sensing Image Super-Resolution},
  author={Xie, Z and Wang, J and Du, Y and others},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={544--558},
  year={2025},
  address={Singapore},
  publisher={Springer Nature Singapore}
}
}""

## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and Recursive generalization transformer for image
super-resolution(RGT)[link](https://github.com/zhengchen1999/RGT).

