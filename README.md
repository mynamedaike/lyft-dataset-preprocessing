# lyft-dataset-preprocessing

Welcome to **L5Kit**. L5Kit is a Python library with functionality for the development and training of *learned prediction, planning and simulation* models for autonomous driving applications.

[Click here for documentation](https://woven-planet.github.io/l5kit)

## 1. Clone the repo

```
git clone https://github.com/mynamedaike/lyft-dataset-preprocessing.git
cd lyft-dataset-preprocessing/l5kit/l5kit
```

## 2. Install L5Kit

```
pip install -e ."[dev]"
```

## 3. Download the training dataset from [LYFT LEVEL 5 OPEN DATA](https://level-5.global/download/)

## 4. Place the train.zarr directory in the sample directory

## 5. Run the script

```
cd ../..
python saveDataToCSV.py
```
CSV files will be generated and saved in the dataset directory.
