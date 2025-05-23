# Module 1: Introduction

Technically means the set of good practices to put ML models in production.

Problem: We will predict taxi services. Such as the duration of the trip.

Usually we can separate a ML process in 3 steps.

1. Design: Do we need ML to solve this problem?
2. Train: We test different models searching which one suits better the data and prediction.
3. Operate: Apply the model to new data.

For our experiment, we have an user, he makes an API request to tell where will he go and then the API will return the duration of the trip.

MLOPs help us to train and deploy the model but also automate the steps in between.

## Optional Lecture: Training a ride duration prediction model

The data corresponds to the taxi data set in parquet format. It is needed to use pyarrow or fastparquet to read the data and tranform it into a pandas dataframe.

* The data can be downloaded from: [Taxi data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
* For the sake of the homework we will use the Yellow taxi data from january and February.

The initial step is to read and preprocess the data based on the question to answer. Finding the columns that will most likely be the most informative.

### One hot encoding

> It allows encoding of categorical values in numerical ones. This method represents each category of a single column to be converted into 1 if the vlaues belongs to that category and 0 otherwise.

## Course Overview

When writing code and reviewing after it generates questions. Do we need certain steps?, how can we simplify or improve the code to make it, modular, reusable and more mature.

When creating models sometimes we need to return to old parameters and remember which set of options were the best.

A way to do this can be on notebooks but notebooks are more experimental and declarative. Whereas, printing the log as a experiment tracker, will help to keep a more clear history of the revisions.

* Model registry: Since we save the models into pickle files that can be read after. It is important to keep the performance of such model. Having a model registry will keep track of each of the models. Usually it goes alongside the experiment tracker (MLFlow).

* ML Pipelines: How can we break down a ML pipeline in several steps like:
  
  * Load and prepare data
  * Vectorize the dataframe
  * Training

* Deployment: The output of a pipeline is a model that we need to take, and put on a ML service to be used in new data or in request.

* Monitoring: Step to verify the performance of a model after a period of time. It is comun that some models lose performance after a while and need to be retrained or updated based on new data.

## MLOps Maturity model

There are 5 levels of maturity in the model.

0. No automation: The way ML is done is simpler, just a jupyter notebook for reserach without bering able to scale.
1. DevOps, no MLOps: We use some practices to deploy a model as webservice, there are some automatization, tests, CICD, and other practices to release models but isolated of ML processes. They are focused only in the releases.
   * Releases are automated
   * Unit and Integration Tests
   * CI/CD
   * Ops Metrics
   * No experiment tracking
   * No reproducibility
   * Data scientist separated from the enginering team
2. Automated training: There is a separate script parametrized that can loads new data and train the model or just a training pipeline. If you have more than 2 models it is suggested to go to this level.
   * Training pipeline
   * Experiment tracking
   * Model registry
   * Low friction deploymenyt
   * DS team with Engineering.
3. Automated deployment: No human involvement in the deplyment process.
   * Easy to deploy model, can be done by API calls to an ML platform.
   * A/B testing, To evaluate which version of a model performs better.
   * Model monitoring
4. Full MLOps Automation: This includes all the previous levels combined.

The level at which your development sets your level depends of the model and the goal. If is a POC then level 0 will suffice and then scales to a new level.

