# C3D Sports1M
## Information

**Reference**:

Tran, Du, et al. *"[Learning Spatiotemporal Features With 3D Convolutional Networks](http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)."* Proceedings of the IEEE International Conference on Computer Vision. 2015.


**Framework used**: *[C3D](https://github.com/facebook/C3D) (Caffe fork)*

**Dataset used to train**: [Sports1M](https://github.com/gtoderici/sports-1m-dataset)


## Description

This model was trained using a modified [version](https://github.com/facebook/C3D)
of BVLC Caffe to support 3-Dimensional Convolutional Networks.
The C3D pre-trained model provided was trained on Sports-1M dataset and can be
used to extract 3D-conv features.

Here are some results from the paper using the C3D features.

| Dataset          | UCF101 | ASLAN       | UMD-Scene | YUPENN-Scene | Object |
| ---------------- | :----: | :---------: | :-------: | :----------: | :----: |
| C3D + linear SVM | 82.3   | 78.3 (86.5) | 87.7      | 98.1         | 22.3   |

If used this model, please refer to the citations on the project [website](http://vlg.cs.dartmouth.edu/c3d/).
