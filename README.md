# runpod_pgcview
![GitHub](https://img.shields.io/github/license/BoMeyering/runpod_pgcview?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/BoMeyering/runpod_pgcview?style=flat-square)
![Docker Image Size](https://img.shields.io/docker/image-size/bmeyering/pgc_view)
![Docker Image Version (tag)](https://img.shields.io/docker/v/bmeyering/pgc_view/0.0.5)






PGCView is an Pytorch based, image analysis pipeline used to measure the proportion of various vegetation classes within experimental field trials evaluating perennial groundcovers (PGC). PGC are perennial grass species such as *Poa bulbosa* and *Poa pratensis* that are planted and established in fields with conventional grain crops like corn and soybeans.

This image analysis pipeline is composed of two parts: an object detection model (EfficientDet architecture) and a semantic segmentation model (DeepLab V3 Plus). The output is a list of the proportion of class pixels within an ROI.

