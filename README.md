## ANDA:  A Novel Data Augmentation Technique Applied to Salient Object Detection
Official code for the ICAR 2019 paper ["ANDA:  A Novel Data Augmentation Technique Applied to Salient Object Detection"](https://arxiv.org/abs/1910.01256)

<p align="center">
<img src="./documentation/flowchart.jpg">
</p>

## REQUIREMENTS:
We recommend the use of conda alternatively miniconda for python environment management. For the scripts, refer to the requirements.txt in the root folder, for the PConvInpainting refer to PConvInpainting/requirements.txt.


## STEP BY STEP USAGE:

- Download MSRA10K at [https://mmcheng.net/msra10k/](https://mmcheng.net/msra10k/)
- Run scripts/protocol.sh PATH ; e.g. MSRA10K_Imgs_GT/Imgs
- Run PConvInpainting/inpaintMSRA10K.py ; --help for parameter instructions
- Run scripts/featureRelated/computeKnn.py ; --help for parameter instructions
- Run scripts/featureRelated/anda.py ; --help for parameter instructions

A bash script for the entire process is available at scripts/run.sh

```
cd scripts
bash run.sh
```

### CITATION:
If you found this code useful for your research, please cite:
```
@INPROCEEDINGS{ruiz2019anda,
author={D. V. {Ruiz} and B. A. {Krinski} and E. {Todt}},
booktitle={2019 19th International Conference on Advanced Robotics (ICAR)},
title={ANDA: A Novel Data Augmentation Technique Applied to Salient Object Detection},
year={2019},
pages={487-492},
keywords={feature extraction;learning (artificial intelligence);neural nets;object detection;video signal processing;image cropping;ANDA technique;labeled salient objects;image inpainting;background information;data augmentation technique;salient object detection context;mean absolute error;F-measure},
doi={10.1109/ICAR46387.2019.8981655},
ISSN={null},
month={Dec},}
```

### DISCLAIMER:

- This is a research code, so compatibility issues might happen.
- The PConvInpainting folder contain code from the [REPOSITORY](https://github.com/MathiasGruber/PConv-Keras).
