# rgb2nir

    • This repository contains a Python 3.4 or higher implementation for the multispectral image 
    registration and using the aligned images in a supervised image translation system. 
    • For multispectral image calibration please follow the instruction in the Micasense Rededge
    documentation (https://github.com/micasense/imageprocessing).
    • For supervised image translation please follow pix2pix model with initial modifications 
    mentioned in the rgb2nir paper. 
    • The dataset folder contains samples of each crop used in our study. TrainA represents RGB images
    and trainB contains NIR couterparts. A random uniform 256 × 256 patch of the RGB image is used as 
    input for the model and it is translated to NIR image with the same size. At inference, we compare 
    the performance of the model with a larger patche of size 512 × 512 as input.
