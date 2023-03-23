# VinDr-RibCXR: A Benchmark Dataset for Automatic Segmentation and Labeling of Individual Ribs on Chest X-rays

This repository contains the training code for our paper entitled "VinDr-RibCXR: A Benchmark Dataset for Automatic Segmentation and Labeling of Individual Ribs on Chest X-rays", which was submitted and under review by [Medical Imaging with Deep Learning 2021 (MIDL2021)](https://2021.midl.io/) 
# Abstract:
We introduce a new benchmark dataset, namely VinDr-RibCXR, for automatic segmentation and labeling of individual ribs from chest X-ray (CXR) scans. The VinDr-RibCXR contains 245 CXRs with corresponding ground truth annotations provided by human experts. A set of state-of-the-art segmentation models are trained on 196 images from the VinDr-RibCXR to segment and label 20 individual ribs. Our best performing model obtains a Dice score of 0.834 (95% CI, 0.810{0.853) on an independent test set of 49 images. Our study, therefore, serves as a proof of concept and baseline performance for future research.\
Keywords: Rib segmentation, CXR, Benchmark dataset, Deep learning.

# Data:
- The dataset contains 245 images and split into 196 training images and 49 validation images. To download the VinDr-RibCXR dataset, please sign our [Data Use Agreement](https://drive.google.com/file/d/1Wr3iI7-OwZHD4eWtCALpRZKvhJuemKAs/view?fbclid=IwAR2lmFoe5JCqpkCVApIc_oXDnldJ21BGpib1PebC3GysrEkjfnqn-Wh2NE8) (DUA) and send the signed DUA to Ha Nguyen (v.hanq3@vinbigdata.com) for obtaining the downloadable link.
- To record the label, we use json file, the json file have this format:
```
{
    "img": {
        "0": "data/train/img/VinDr_RibCXR_train_000.png",
        ....
    }
    "R1": {
        "0":[
            {
                "x": 1027.8114701859,
                "y": 470.688105676
            },
            ....
        ]
    }
    ....
}        
```
- To view the label, you can go to vinlab (https://lab.vindr.ai) or use the notebook visualize.ipynb
- For further description, please go to this (https://vindr.ai/datasets/ribcxr)
# Setting:
We use Pytorch 1.7.1 for this project

# Model:
In this work, we train 4 baselines model, that is vanila U-net and U-net, FPN and U-net plus plus with imagenet pretrained encoder efficientet B0 from (https://github.com/qubvel/segmentation_models.pytorch)
# Train model by your self:
- We provide 4 configs file conresponding to 4 setting in our model, you can also write your config for your self
- You need to put the data like this:
```
data
├── train
|   ├──img/
|   └──Vindr_RibCXR_train_mask.json
└── val
    ├──img/
    └──Vindr_RibCXR_val_mask.json
```
- For training use this command:
```
python main.py --config/cvcore/multi_unet_b0_diceloss.yaml
```

# Result:


| Model                      | Dice             | 95% HD                 | Sensitivity      | Specificity      |
|----------------------------|------------------|------------------------|------------------|------------------|
| U-Net                      | .765 (.737-.788) | 28.038 (22.449-34.604) | .773 (.738-.791) | .996 (.996-.997) |
| U-Net w. EfficientNet-B0   | .829 (.808-.847) | 16.807 (14.372-19.539) | .844 (.818-.858) | .998 (.997-.998) |
| FPN w. EfficientNet-B0     | .807 (.773-.824) | 15.049 (13.190-16.953) | .808 (.773-.828) | .997 (.997-.998) |
| U-Net++ w. EfficientNet-B0 | .834 (.810-.853) | 15.453 (13.340-17.450) | .841 (.812-.858) | .998 (.997-.998) |

# Acknowledgements:
This research was supported by the Vingroup Big Data Institute. We are especially thankful
to Tien D. Phan, Dung T. Le, and Chau T.B. Pham for their help during the data annotation
process