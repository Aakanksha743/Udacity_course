# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.I tried to optimize the hyperparameters using
Hyperdrive.,then used AutoML to find optimal model using the same dataset to compare the results
This model is then compared to an Azure AutoML run.
Step 1: Set up Training Script 'train.py' for building model using Bankmarketing dataset.
Step 2: Set up udacity-project.ipynb notebook to find best hyperparameters of the  sklearn model using hyperdrive.
Step 3: Use AutoML on the same dataset to find another optimal model and compare the results.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
The dataset contains bank marketing data of bank institution from direct marketing campaigns.The Classification goal was to
predict column 'y' and detrmine whether a customer will subscribe for loan or not.
![image](https://github.com/Aakanksha743/Udacity_course/assets/151511734/6ee88a6b-12a3-4e7e-afdd-17e496ef573e).I have used a simple logistic regression model,chosing more complex model might help improve the accuracy.


**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The Best performing model was Hyperdrive Model with {'run_id': 'HD_eca738bc-f660-4c7b-920f-36b6ab6b9b22_10'} with 'best_primary_metric'(Accuracy): 0.9157814871016692.
It is derived from sklearn pipeline,whereas in AutoML Model({'runId': 'AutoML_5520758b-160e-4815-b592-806fab770c97),the Best Run Metrics Best run  was {'Regularization Strength:': 10.0, 'Max iterations:': 300, 'Accuracy': 0.9176024279210926} and the algorithm used was Voting Assemble.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
1.The dataset was bankmarketing dataset retrieved from the given url from TabularDataFactory class.
2.In train.py ,did some data preprocessing using clean_data function.
3.Defined sklearn model and fit it.
4.Define ScriptRunConfig to pass to Hyperdriveconfig and used RandomparameterSampling with hyperpramaets with maximum number of iterations (50,100,200,300) and inverse regularization paramete(C).
We define a HyperDrive Config using the ScriptRunConfig, parameter sampler and an early termination policy. Then, we submit the experiment.

After the run is completed, we found that the best model has {'Regularization Strength:': 10.0, 'Max iterations:': 300, 'Accuracy': 0.9176024279210926}

**What are the benefits of the parameter sampler you chose?**
Defined parameter sampler as
![image](https://github.com/Aakanksha743/Udacity_course/assets/151511734/a19de5e8-0155-4105-8f2c-3d2cbc976682)

In Random sampling hyperparameters are randomly selected from defined search space.'C' is teh regularization while 'max_iter' is the maximum number of iterations.Hyperdrive can try all possible combinations from search space and find the best optimal model.I chose RandomHyperparameter because it is faster and supports early termination of low performance runs.


**What are the benefits of the early stopping policy you chose?**
An early stopping policy is uded to automatically terminate poorly performing runs thus improving computational efficiency and prevent long runs from using resources.
I used a Bandit policy for early termination with the parameters evaluation_interval=2 and slack_factor=0.1.
![image](https://github.com/Aakanksha743/Udacity_course/assets/151511734/b16e960d-8c57-49f3-9de7-01d7be693683)
Evaluation_interval - This is optional and represents the frequency for applying the policy.Each time the training scripts logs the primary metric counts as one interval.
Slack - The amount of slack allowed with respect to best performing training runs.This factor specifies teh slack as ratio.If budget is not an issue,we could use grid parameter sampling to exhaustively search over the search space. or Bayesian Hyperparameter sampling to explore hyperparameters over the search space.

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish and this is the reason I chose it


## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
To use AutoML in Azure,We first define AutoMLConfig in which we define compute_target(compute cluster),experiment timeout minutes,dataset,primary metric,max_concurrent_iterations and iterations.
AzureML tried different model RandomForest,XGBoost,BoostedTrees,VotingEnsemble etc... .out of which the best model was Prefittedsoftvoting classifier. The best model had the following metrics.
norm_macro_recall': 0.4371276755154334,
1.'AUC_micro': 0.9807799521507965,
2. 'precision_score_weighted': 0.9059900421779148,
3. 'accuracy': 0.9148710166919576,
4.'log_loss': 0.18477291615729186,
5. 'recall_score_micro': 0.9148710166919576,
6. 'average_precision_score_macro': 0.8254781523696038,
7. 'f1_score_micro': 0.9148710166919576,
8. 'f1_score_macro': 0.751684758824199,
9. 'average_precision_score_weighted': 0.9557548350268407,
10. 'average_precision_score_micro': 0.9816152057287895,
11. 'f1_score_weighted': 0.9078713376714261,
12. 'recall_score_macro': 0.7185638377577167,
13. 'precision_score_macro': 0.8051812644263784,
14. 'balanced_accuracy': 0.7185638377577167.

Parameters generated by AutoML Best Model
learning_rate=0.1,
max_depth=-1,
min_child_samples=20,
min_child_weight=0.001,
                                                                             
l1_ratio=0.8979591836734693
learning_rate='constant',
loss='modified_huber'
max_iter=1000,
 n_jobs=1,
 penalty='none',
 power_t=0.6666666666666666,
 random_state=None,
tol=0.01)

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
No big difference between Hyperdrive and AutoML Model in terms of accuracy.In terms of architecture and various metric( like AUC_weighted(more fit for imbalanced data),AUC Micro,recall_score_micro..) ,AutoML is superior and we can get a better model.I chose Accuracy as the primary metric and enabled ONNX_compatible models.
The best model generated by AutoML is Voting Ensemble

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
We can try more hyperparameters for the improvement of the model.Try different sampling methods.Another parameter we can improve is number of cross validations.It is a process of taking many subsets of data and training a model on each subset.Higher the number of cross validations higher is teh accuracy,but in that case cost will be increased because i have to increase experiment_time_out_minutes.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
![image](https://github.com/Aakanksha743/Udacity_course/assets/151511734/1d9cfc31-e09c-412b-b39f-e8ca88c8b480)

