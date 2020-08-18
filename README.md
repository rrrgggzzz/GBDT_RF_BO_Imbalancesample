# GBDT_RF_BO_Imbalancesample
The first step is using the “Randomundersample.py” and “BorderlineSMOTE.py” for sample processing.
The second step is to use “RF.py” and “GBDT.py” to run and validate the GBDT and RF models using the default hyperparameters.
The third step is using “RF_Bayes.py” and “GBDT_Bayes.py” to perform the Bayesian optimization of the GBDT and RF models.
Finally, enter the optimized hyperparameters into the “RF_Bayes_Run.py” and “GBDT_Bayes_Run.py” to run and validate the GBDT_B and RF_B models.
