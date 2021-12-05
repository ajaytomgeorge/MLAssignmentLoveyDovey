# Supervised Learning Assignment: Lovey Dovey

All updated files can be found at https://github.com/ajaytomgeorge/MLAssignmentLoveyDovey.
Please download files from git repo stated above to download latest algorithms added

## Goal
We are training a binary classifier, attempting to get good generalization performance. 

## Requirements
Have python 3.7 or higher installed on your computer along with pip

## Installation

Copy train.txt and test.txt into the directory
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install run the project.

run
```bash
pip install - requirements.txt
```

## To generate test-o.txt

```bash
python .\run_model.py
```

## Re-train the Best classifer - MLP

 Multi Layer Percpetron classifier from scratch 
MLP was selected as the best classifier with 72% accuracy


```bash
python .\train_model_mlp.py
```


## Other Classifier to try

1. Logistic Regression

```bash
python train_model_logistic_regression.py
```

2. K means - Unsupervised Algorithm
```bash
python train_model_kmeans.py
```
3. Random Forest
```bash
python random_forest.py
```
4. Support Vector Machine
```bash
python random_forest.py
```
5. Convolutional Neural Network
```bash
Execute the attached jupyter notebook cnn.ipynb
```
