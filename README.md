# semantic-segmentation
Deep Learning Model for performing semantic segmentation on cityscapes dataset

## Installation:
```bash
ssh ucsd_username@dsmlp-login.ucsd.edu
launch-scipy-ml-gpu.sh
pip install virtualenv

git clone https://github.com/kj141/semantic-segmentation.git
cd semantic-segmentation
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Directory
```
.
+-- README.md
+-- LICENSE
+-- requirements.txt
+-- dataloader.py
+-- utils.py
+-- basic_fcn.py
+-- starter.ipynb
+-- Data
|   +-- train.csv
|   +-- test.csv
|   +-- val.csv
+-- venv
|   +--
+-- models
|   +--
```
