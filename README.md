# OpenLPR Plate Detection
The Car License Plate Detection component of OpenLPR project.

This repository is forked from [OpenLPR project](https://github.com/faisalthaheem/open-lpr)

# Deploying
```
pip install sagemaker-studio-image-build

sm-docker build .
```


# Testing locally
Download model from https://johnespe-open-lpr-plate-detection-model.s3.amazonaws.com/SSD.pth 

Install peer dependencies
```
pip install -r requirements.txt
```

Run
```
from standalone import StandAloneInference

StandAloneInference(model_path="SSD.pth").process_from_disk("photo.jpg")
```