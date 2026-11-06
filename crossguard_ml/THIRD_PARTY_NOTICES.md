# Attribution and provenance

## Detector

Architecture and checkpoint: torchvision SSDLite320 MobileNetV3 Large, COCO_V1.
Source: https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth
Architecture: https://github.com/pytorch/vision
The upstream torchvision BSD-3-Clause code license is included as `models/TORCHVISION_LICENSE.txt`. Pretrained weights derive from COCO; check upstream model/data terms for uses beyond this PoC. The project MIT license does not relicense third-party weights or images.

Original research includes SSD, Searching for MobileNetV3, and MobileNetV2: Inverted Residuals and Linear Bottlenecks; see the torchvision architecture documentation for references.

## Included public smoke-test photos

1. `samples/crosswalk_cc0.jpg`: Peter Miranda, “Pedestrians using crosswalk”, 6 June 2015. Wikimedia Commons identifies the image as CC0, originally published on Unsplash before its 2017 license change. Source: https://commons.wikimedia.org/wiki/File:Pedestrians_using_crosswalk_(Unsplash).jpg
2. `samples/bus_india_cc0.jpg`: Soumava2002, “V-1 standing at Airport Bus Terminus”, 28 September 2025. CC0. Source: https://commons.wikimedia.org/wiki/File:V-1_standing_at_Airport_Bus_Terminus.jpg

CC0: https://creativecommons.org/publicdomain/zero/1.0/
Included copies were decoded, orientation-normalized by OpenCV and resized to a maximum side of 1280 pixels. Annotated versions add model predictions and a count banner. These images were used only for inference smoke tests, not new training. They are not organiser-provided material and cannot substitute for required organiser-input evidence.
