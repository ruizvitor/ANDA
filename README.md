## ANDA:  A Novel Data Augmentation Technique Applied to Salient Object Detection
Official code for the ICAR 2019 paper "ANDA:  A Novel Data Augmentation Technique Applied to Salient Object Detection"

<p align="center">
<img src="./documentation/flowchart.jpg">
</p>



## STEP BY STEP USAGE

- Download MSRA10K at [https://mmcheng.net/msra10k/](https://mmcheng.net/msra10k/)
- Run scripts/protocol.sh PATH , e.g. MSRA10K_Imgs_GT/Imgs
- Run PConvInpainting/inpaintMSRA10K.py , --help for parameter instructions
- Run scripts/featureRelated/computeKnn.py, --help for parameter instructions
- Run scripts/featureRelated/anda.py, --help for parameter instructions

A bash script for the entire process is available at scripts/run.sh

```
cd scripts
bash run.sh
```

If you found this code useful for your research, please cite:
```
TO DO
```

### DISCLAIMER:

- This is a research code, so compatibility issues might happen.
- The PConvInpainting folder contain code from the [REPOSITORY](https://github.com/MathiasGruber/PConv-Keras).
