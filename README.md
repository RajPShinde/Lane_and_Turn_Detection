# Lane & Turn Detection
[![License MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](Colour_Segmentation-Gaussian_Mixture_Model-and-Expectation_Maximization/blob/master/LICENSE)

## Authors
* **Raj Prakash Shinde** [GitHub](https://github.com/RajPShinde)
* **Shubham Sonawane** [GitHub](https://github.com/shubham1925)
* **Prasheel Renkuntla** [GitHub](https://github.com/Prasheel24)

## Description
The project contains the implementation of a pipeline for Lane and Turn Detection using histogram of lanes approach. The video is converted in HLS colour space to filter out the lanes. Furter the Warp perspective and the Homograpgy methods are Implemented from scratch to  create overlays on lanes

## Output
<img src="/Lane_Detection.png"/>

## Dependencies
* Ubuntu 16
* Python 3.7
* OpenCV 4.2
* Numpy
* copy
* sys
* argparse

## Run
To run the lane detection on challenge video-
```
python3.7 Lane_Detection.py
```
## Reference
* https://stackoverflow.com/questions/19890054/how-to-sharpen-an-image-in-opencv
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
* https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
* https://medium.com/@rndayala/image-histograms-in-opencv-40ee5969a3b7
* https://answers.opencv.org/question/193276/how-to-change-brightness-of-an-image-increase-or-decrease/
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
* https://www.programcreek.com/python/example/89353/cv2.createCLAHE
* http://amroamroamro.github.io/mexopencv/opencv/clahe_demo_gui.html
* https://en.wikipedia.org/wiki/Kernel_(image_processing)
* https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/


