# US Housing Regression Task using Simple Neural Network vs. XGBoost, LightGBM
## Overview
This repository presents a machine learning project comparing traditional machine learning (ML) with deep learning (DL) algorithms for predicting California housing prices. 
The project compares different models:
- **XGBoost** – a small custom convolutional model  
- **LightGBM** – a deeper custom network  
- ***Neural Networks** – pretrained on ImageNet and fine-tuned for CIFAR-10  

To ensure precise model efficiency, we utilize Nested Cross-Validation for hyperparameters tuning and implement consistent evaluation at outer folds. 
Pipeline integration is also applied to prevent data leakage.


## Description
This project is a regression task, focusing on several aspects on:
- *Nested Cross-Validation* for unbiased model evaluation
- Hyperparameter optimization using *RandomizedSearchCV*
- *Statistical significance testing* for model comparisons
- Model *interpretability* using SHAP values to investigate feature importance
- Feature engineering and preprocessing:
  - PCA for geographic data using custom sklearn transformers
  - Interaction features and feature removal
  - Log transformation for right-skewed target values
  - VarianceThreshold: removes low-variance features

## Dataset
 The dataset originally comes from the 1990 U.S. census, which has:
- Features: 8 numerical features (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- Target: Median house value
- Instances: 20,640 housing blocks

## Result
Some insights:
- It seems that Median Income (MedInc) is the strongest predictor.
- House age have moderate negative correlation with price.
- Traditional ML outperforms deep learning (DL), providing some rooms of improvements for DL.
<table>
  <tr>
    <th>model</th>
    <th>MSE</th>
    <th>RMSE</th>
    <th>MAE</th>
    <th>Params</th>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>0.030816</td>
    <td>0.175537</td>
    <td>0.12809</td>
    <td>{'regressor__n_estimators': 500, 'regressor__m...</td>
  </tr>
  <tr>
    <td>LightGBM</td>
    <td>0.030792</td>
    <td>0.175466</td>
    <td>0.12815</td>
    <td>{'regressor__n_estimators': 10000, 'regressor_...</td>
  </tr>
  <tr>
    <td>DeepLearning</td>
    <td>0.039986</td>
    <td>0.199966</td>
    <td>0.15083</td>
    <td>{'hidden_dim': 64, 'dropout': 0.0, 'lr': 0.001}</td>
  </tr>
</table>


## Quick Start
- Clone this repository: git clone https://github.com/ignsagita/regression-nn-housing.git cd regression-nn-housing
- Install dependencies pip install -r requirements.txt
- Recommended Setup: For the best experience, **run this notebook on [Google Colab](https://colab.research.google.com/)** 
- In Colab, **enable GPU support** by going to: `Runtime > Change runtime type > Hardware accelerator > GPU`

## Future Enhancement
- Ensemble methods: Combine predictions from multiple models
- Advanced neural architectures, such as attention mechanisms
- Bayesian optimization: More sophisticated hyperparameter tuning
- Additional models: Random Forest, Support Vector Regression (SVR)

---
