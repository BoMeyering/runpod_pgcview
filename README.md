# runpod_pgcview
PGCView is an Pytorch based, image analysis pipeline used to measure the proportion of various vegetation classes within experimental field trials evaluating perennial groundcovers (PGC). PGC are perennial grass species such as *Poa bulbosa* and *Poa pratensis* that are planted and established in fields with conventional grain crops like corn and soybeans.

This image analysis pipeline is composed of two parts: an object detection model (EfficientDet architecture) and a semantic segmentation model (DeepLab V3 Plus). The output is a list of the proportion of class pixels within an ROI.

