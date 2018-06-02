What is it?

This repository contains an MATLAB implementation of the following paper:
Shanmin Pang, Jin Ma,  Jianru Xue, Jihua Zhu, and Vicente Ordonez, Deep Feature Aggregation with Heat Diffusion for Image Retrieval,arXiv:1805.08587.

If you find this package is useful, please kindly cite our paper as follows
@article{pang2018image,
  title={Image Retrieval using Heat Diffusion for Deep Feature Aggregation},
  author={Pang, Shanmin and Ma, Jin and Xue, Jianru and Zhu, Jihua and Ordonez, Vicente},
  journal={arXiv preprint arXiv:1805.08587},
  year={2018}
}

This code implements

a) HeW image representation

b) Image/object retrieval based on HeW on public datasets: Oxford5k, Paris6k and Holidays.

The code is written by: Shanmin Pang (pangsm@xjtu.edu.cn). If you have any questions, please contact Shanmin Pang.

Setup
Dependencies

MatConvNet v1.0-beta18 or above (http://www.vlfeat.org/matconvnet/).

Models:
Two models used in our experiment are as follows:

Vgg16: imagenet-matconvnet-vgg-verydeep-16 (http://www.vlfeat.org/matconvnet/models/imagenet-matconvnet-vgg-verydeep-16.mat)
SiaMAC: retrievalSfM120k-siamac-vgg (http://cmp.felk.cvut.cz/cnnimageretrieval/networks/retrieval-SfM-120k/retrievalSfM120k-gem-vgg.mat)

Dataset

Oxford5k: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
Paris6k: http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
INRIA Holidays: https://lear.inrialpes.fr/~jegou/data.php#holidays
Flickr 100k: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/flickr100k.html

Execution
Extract features: See 'feature_extract.m' file, please change the pathname to the correct folder where you store images.

'train_oxford.m' file gets the representations of training images with features extracted by SiaMAC
'test_holiday.m' file gets the representations of test images with features extracted by SiaMAC
'search.m' file shows results of our method HeW on the Holidays dataset with features extracted by SiaMAC.