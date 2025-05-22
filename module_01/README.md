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

* ML Pipelines: How can we break down a ML pipeline in several steps, 

