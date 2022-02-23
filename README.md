# Predicting the spread of fixed income instruments

The goal of the project is to create a model predicting the Option Adjusted spread (OAS) of a bond based on factors contributing to the spread. 
It is widely known that the riskier the bond (low credit rating) the higher the fee (OAS) it is paying to its buyers. Investors get compensated for the risk.
However, there are other aspects that play a role here such as how big the issuer of the bond is (e.g. given the same credit rating a big issuer is less likely to default compared to a small one) or the country of issuance.
Additionally, there are also stochastic factors that play a role. Therefore, by using some of the characteristics of the bonds we can predict the spread, hence determine if we should by a bond or not.

## Project Set Up and Installation
There are no special installation requirements. The project is executed in two notebooks, on Azure Automl, using Python 3.6. 
There are a number of packages required, all of them can be found in the conda_dependencies document and can be easily installed using pip.
## Dataset

### Overview
The data used is a so-called fixed income data. It is a list of bonds with their properties such as: Credit rating, size of the issuer, country of risk, industry etc.
The set is quite standard for for the fixed income area.
### Task
The task is to use the characteristics of the bond mentioned above in order to explain (predict) the Option Adjusted Spread of a bond.
### Access
The data was registered in Azure automl. Thereafter, using the Workspace object (ws) the data was loaded using ws.datasets['df_bonds'] (see automl notebook).
## Automated ML
For the automl project the following settings were used:
1. Task - regression - appropriate for predicting continuous numeric variable I have, OAS.
2. experiment_timeout_minutes - 20 - given the limited time in the lab I wanted to also limit the time of experiment. The minimum is 15 minutes and the decision was taken to increase it slightly such that more Automl models will be tested.
3. max_concurrent_iterations - 5 - a higher number would be better, however that would increase the time for completing each model within the experiment. With this number I was trying to find the balance between the quality and the quantity of the automl models.
4. primary_metric - r2_score - one of the standard metrics, appropriate for regression and continuous numeric variable I am trying to predict.
5. an early stopping was also enabled in order to increase the number of models being produced within the 20 minutes experiment time.
Thereafter, the model was trained on a cluster created by me.
### Results
The accuracy of the trained model (the best one) is good, with the r2 metric of 0.19. The best model based on the main metric is Extremely Randomized Trees classifier in combination with a Sparce Normalizer.
There are certainly a number of improvements possible.
1. For the sake of efficiency, due to the time constrains, I have chosen a relatively small dataset with a relatively small number of explanatory variables.
Adding more variables and expanding the dataset will certainly add value.
2. The experiment timeout should be increased. 20 minutes is a relatively short time for a very advanced machine learning model. Increasing this time would allow one to possibly find a more comprehensive model.
3. The same as above is valid for the number of the maximum concurrent iteration.
As mentioned above, the only reason for not choosing better parameters is the time constrains.

## Hyperparameter Tuning
For the hyperparameter tuning I have chosen a Stochastic Gradient Descent (SGD) linear model. 
In this method, during the training, the model is updated along the way with a decreasing learning rate. 
This model was chosen as an advance alternative for the linear regression model, with the main advantage here being the training methodology used.
The model is appropriate for predicting the continuous numeric variable, which is the case for OAS, hence fits perfectly the purpose of the project.
The two main parameters for the SGD method are "alpha" and the "l1_ratio". The former is a constant that multiplies the regularization term. The higher the value, the stronger the regularization.
The latter is the Elastic Net mixing parameter, and is only used if penalty is ‘elasticnet’ (which was chosen to be the parameter for the model).
Both parameters range within 0.1 and 1.
For the rationale behind the choice of the early termination policy and the parameter sampling method see the jupyter notebook.

### Results
The results for the best model from the hypermarameter tuning are much worse than for the ML model from Automl.
The main reasons for that is the number of models that were tested. I restricted the maximum number of runs is 30, which is quite low given the two-dimensional nature of the model.
The best SGD model was found to have the following parameters: alpha is 0.74, l1_ratio is 0.95
There are a number of improvements to be made. Those will also bring the accuracy of the model to an acceptable level:
1. For the sake of efficiency, due to the time constrains, I have chosen a relatively small dataset with a relatively small number of explanatory variables.
Adding more variables and expanding the dataset will certainly add value.
2.The number of maximum total runs in HyperDriveConfig should be increased from 30. 30 is a low number given there are two parameters to optimize. 
The only reason for choosing such a low number is the time constrain.
3. One can lift the early termination policy in order to decease the probability that a potentially good model will stop on an early stage but will increase the compute time.
4. A grid approach for testing the hyperparameters will be more robust than the uniform random sampling currently used. This, however, again will increase the compute time.

## Model Deployment
The best model that was deployed is an Automl model. The prediction error was much lower that that of the model from hyperparameter tuning. The deployed model has the following characteristics:
run algorithm: VotingEnsemble, ensembled algorithms: ['XGBoostRegressor', 'XGBoostRegressor', 'LightGBM', 'LightGBM', 'LightGBM', 'LightGBM', 'ElasticNet', 'XGBoostRegressor', 'XGBoostRegressor', 'XGBoostRegressor']
The model was deployed via the endpoint, hence the model can be easily consumed. Since in Jupyter notebook I already had the Webservice returned after I deployed the model (aci_service=Model.deploy(ws,'valeriywebservice7',[model],inference_config,aciconfig)),
I used that webservice to send the test data to the the endpoint and receive the result (aci_service.run(input_data=test_sample)). Note that the data should be converted to the json format.
Obviously, there is also REST endpoint generated after the model has been successfully deployed (see snapshot), which can be used for consumption of the model at any time.

## Screen Recording
https://youtu.be/tFl8NR1gL98