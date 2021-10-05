# HEALTH-INSURANCE-CROSS-SELL-PREDICTION
Predicting whether the customer would be interested in buying Vehicle insurance or not.



---

## Table of Content
  * [Abstract](#abstract)
  * [Problem Statement](#problem-statement)
  * [Data Description](#data-description)
  * [Project Outline](#project-outline)
    - 1 [Data Wrangling](#data-wrangling)
    - 2 [Standardization](#standardization)
    - 3 [EDA](#eda)
    - 4 [Text Pre-processing](#text-pre-processing)
    - 5 [Encoding categorical values](#encoding-categorical-values)
    - 6 [Feature Selection](#feature-selection)
    - 7 [Model Fitting](#model-fitting)
    - 8 [Hyper-parameter Tuning](#hyper-parameter-tuning)
    - 9 [Metrics Evaluation](#metrics-evaluation)
    - 10 [Feature Importance - SHAP Implementation](#feature-importance-shap-implementation)
  * [Conclusion](#run)
  * [Reference](#reference)

---


# HEALTH-INSURANCE-CROSS-SELL-PREDICTION
Predicting whether a customer would be interested in buying Vehicle Insurance so that the company can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue.


---

## Table of Content
  * [Abstract](#abstract)
  * [Problem Statement](#problem-statement)
  * [Data Description](#data-description)
  * [Project Outline](#project-outline)
    - 1 [Data Wrangling](#data-wrangling)
    - 2 [Standardization](#standardization)
    - 3 [EDA](#eda)
    - 4 [Text Pre-processing](#text-pre-processing)
    - 5 [Encoding categorical values](#encoding-categorical-values)
    - 6 [Feature Selection](#feature-selection)
    - 7 [Model Fitting](#model-fitting)
    - 8 [Hyper-parameter Tuning](#hyper-parameter-tuning)
    - 9 [Metrics Evaluation](#metrics-evaluation)
  * [Conclusion](#run)
  * [Reference](#reference)

---


# Abstract
Will add abstract here

# Problem Statement
Our client is an Insurance company that has provided Health Insurance to its customers. Now they need the help in building a model to predict whether the policyholders (customers) from the past year will also be interested in Vehicle Insurance provided by the company.

An insurance policy is an arrangement by which a company undertakes to provide a guarantee of compensation for specified loss, damage, illness, or death in return for the payment of a specified premium. A premium is a sum of money that the customer needs to pay regularly to an insurance company for this guarantee.

*Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimize its business model and revenue.*


# Data Description
We have a dataset which contains information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc. related to a person who is interested in vehicle insurance.
We have 381109 data points available.


| Feature Name | Type | Description |
|----|----|----|
|id| (numeric) |Unique identifier for the Customer.|
|Age |(numeric)| Age of the Customer.|
|Gender |(string, dichotomous)|Gender of the Custome|
|Driving_License |(numeric, dichotomous)|0 for customer not having DL, 1 for customer having DL.|
|Region_Code |(numeric, Nominal)|Unique code for the region of the customer.|
|Previously_Insured| (numeric, dichotomous)| 0 for customer not having vehicle insurance, 1 for customer having vehicle insurance.|
|Vehicle_Age| (numeric, nominal)| Age of the vehicle.|
|Vehicle_Damage | (numeric, dichotomous)| Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.|
|Annual_Premium| (numeric)| The amount customer needs to pay as premium in the year.|
|Policy_Sales_Channel| (numeric, nominal)| Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.|
|Vintage |(numeric)|Number of Days, Customer has been associated with the company.|
|**Response** (Dependent Feature)|(numeric, dichotomous)| 1 for Customer is interested, 0 for Customer is not interested.|


----

# Project Outline

## 1. Data Wrangling
After loading our dataset, we observed that our dataset has 381109 rows and 12 columns. We applied a null check and found that our data set has no null values. Further, we treated the outliers in our dataset using a quantile method.
![image](https://user-images.githubusercontent.com/35359451/135953638-5ed37a2c-faaf-4c09-a276-4afc8e345a02.png)

### Outlier Treatment
![image](https://user-images.githubusercontent.com/35359451/136047262-8aa2ad3e-df50-4dfc-91e9-041ae7fea9ed.png)
--


## 2. Normalization
After outlier treatment, we observed that the values in the numeric columns were of different scales, so we applied the min-max scaler technique for feature scaling and normalization of data.

## 3. EDA
In Exploratory Data Analysis, firstly we explored the 4 numerical features: Age, Policy_Sales_Channel, Region_Code, Vintage. Further, we categorized age as youngAge, middleAge, and oldAge and also categorized policy_sales_channel and region_code. From here we observed that customers belonging to the youngAge group are less interested in taking vehicle insurance. Similarly, Region_C, Channel_A have the highest number of customers who are not interested in insurance. From the vehicle_Damage feature, we were able to conclude that customers with vehicle damage are more likely to take vehicle insurance. Similarly, the Annual Premium for customers with vehicle damage history is higher.

## 4. Encoding categorical values
We used one-hot encoding for converting the categorical columns such as 'Gender', 'Previously_Insured','Vehicle_Age','Vehicle_Damage', 'Age_Group', 'Policy_Sales_Channel_Categorical', 'Region_Code_Categorical' into numerical values so that our model can understand and extract valuable information from these columns.

## 5. Feature Selection
At first, we obtained the correlation between numeric features through Kendall’s Rank Correlation to understand their relation. We had two numerical features, i.e. Annual_Premium and Vintage. 
For categorical features, we tried to see the feature importance through Mutual Information.  It measures how much one random variable tells us about another.
|||
|----|----|
| ![image](https://user-images.githubusercontent.com/35359451/135954415-825af5ec-b9cc-4729-b602-ccc456d88244.png) | ![image](https://user-images.githubusercontent.com/35359451/135954442-a80364b7-652c-42ee-937e-ed69ee52960d.png)|
|||


## 6. Model Fitting
For modeling, we tried the various classification algorithms like:

* Decision Tree
* Gaussian Naive Bayes
* AdaBoost Classifier
* Bagging Classifier
* LightGBM
* Logistic Regression

## 7. Hyperparameter Tuning
Tuning of hyperparameters is necessary for modeling to obtain better accuracy and to avoid overfitting. In our project, we used the GridSearchCV, RandomizedSearchCV, and HalvingRandomizedSearchCV techniques.
![Screenshot from 2021-10-05 20-40-27](https://user-images.githubusercontent.com/35359451/136051125-b4c12fab-aa31-4908-af9e-040d14a81794.png)
![image](https://user-images.githubusercontent.com/35359451/135954494-ed31cdfe-711b-4e41-b4ff-57fe79812daa.png)


## 8. Metrics Evaluation
We used some of the metrics valuation techniques like Accuracy Score, F1 Score, Precision, Recall, Log Loss, to obtain the accuracy and error rate of our models before and after hyperparameter tuning.
![image](https://user-images.githubusercontent.com/35359451/135954471-0ef43329-7b54-456a-823a-fd8c92d4a972.png)



# Algorithms
## i. Decision Tree 
Decision Trees are non-parametric supervised learning methods, capable of finding complex non-linear relationships in the data. Decision trees are a type of algorithm that uses a tree-like system of conditional control statements to create the machine learning model. A decision tree observes features of an object and trains a model in the structure of a tree to predict data in the future to produce output.
For classification trees, it is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

## ii. Gaussian Naive Bayes
Gaussian Naive Bayes is based on Bayes’ Theorem and has a strong assumption that predictors should be independent of each other. For example, Should we give a Loan applicant depending on the applicant’s income, age, previous loan, location, and transaction history? In real-life scenarios, it is most unlikely that data points don’t interact with each other but surprisingly Gaussian Naive Bayes performs well in that situation. Hence, this assumption is called class conditional independence.

## iii. AdaBoost Classifier
Boosting is a class of ensemble machine learning algorithms that involve combining the predictions from many weak learners. A weak learner is a very simple model, although has some skill on the dataset. Boosting was a theoretical concept long before a practical algorithm could be developed, and the AdaBoost (adaptive boosting) algorithm was the first successful approach for the idea.
The AdaBoost algorithm involves using very short (one-level) decision trees as weak learners that are added sequentially to the ensemble. Each subsequent model attempts to correct the predictions made by the model before it in the sequence. This is achieved by weighing the training dataset to put more focus on training examples on which prior models made prediction errors.

## iv. Bagging Classifier
A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

## v. LightGBM 
Light GBM is a gradient boosting framework that uses tree-based learning algorithms. Light GBM grows trees vertically while other algorithms grow trees horizontally meaning that Light GBM grows trees leaf-wise while other algorithms grow level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, a Leaf-wise algorithm can reduce more loss than a level-wise algorithm. Light GBM is prefixed as ‘Light’ because of its high speed. Light GBM can handle the large size of data and takes lower memory to run.

## vi. Logistic Regression
Logistic regression is named for the function used at the core of the method, the logistic function.

The logistic function, also called the sigmoid function, was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.

# Metrics Evaluation
## i. Confusion Matrix
## ii. Accuracy
## iii. Precision
## iv. Recall
## v. F1-Score
## vi. ROC-AUC Score
## vii. Log Loss


# Hyperparameter Tuning
## i. GridSearchCV
## ii. RandomizedSearchCV
## iii. HalvingRandomizedSearchCV


# Challenges Faced

# Conclusion




![Screenshot from 2021-10-05 20-40-27](https://user-images.githubusercontent.com/35359451/136051125-b4c12fab-aa31-4908-af9e-040d14a81794.png)


---
![image](https://user-images.githubusercontent.com/35359451/136047262-8aa2ad3e-df50-4dfc-91e9-041ae7fea9ed.png)

![image](https://user-images.githubusercontent.com/35359451/135953638-5ed37a2c-faaf-4c09-a276-4afc8e345a02.png)
![image](https://user-images.githubusercontent.com/35359451/135953738-2a26a269-cb52-4f15-bde3-368835fe7789.png)
![image](https://user-images.githubusercontent.com/35359451/135953802-b15ecc42-8a54-4c3f-9a8b-41bcc2bc7a6a.png)
![image](https://user-images.githubusercontent.com/35359451/135953853-0a84b0b7-c004-49a7-8fd2-802b27d3e63d.png)
![image](https://user-images.githubusercontent.com/35359451/135953896-713f8524-e774-4514-acf6-11e335cea054.png)
![image](https://user-images.githubusercontent.com/35359451/135953918-4ad34baa-3a67-48b9-a72d-fd5306dca5a6.png)
![image](https://user-images.githubusercontent.com/35359451/135953930-41da0ad3-32c1-41f7-9658-c362290d9874.png)
![image](https://user-images.githubusercontent.com/35359451/135953957-35f79545-6e83-4849-a048-373bdbee7189.png)
![image](https://user-images.githubusercontent.com/35359451/135953982-4e0520db-7344-4250-9f71-4b27cb9a093f.png)
![image](https://user-images.githubusercontent.com/35359451/135954015-636139ed-3aeb-46a3-b289-ecb1e33c511d.png)
**![image](https://user-images.githubusercontent.com/35359451/135954047-b00e3a1c-47f9-405e-94ff-b1afdb4ddddc.png)
![image](https://user-images.githubusercontent.com/35359451/135954074-791c594a-95b7-47ea-97d2-40b702c3115c.png)
![image](https://user-images.githubusercontent.com/35359451/135954162-b3312ca8-6ea9-4079-8f85-d45c1a1085a5.png)
![image](https://user-images.githubusercontent.com/35359451/135954200-1f684681-0990-457e-b65e-d96cde52fa28.png)
![image](https://user-images.githubusercontent.com/35359451/135954212-4f321a9a-6151-4848-a67a-88a8bede28da.png)
![image](https://user-images.githubusercontent.com/35359451/135954221-f52fff61-35f3-40d3-8525-4c2b60d36a80.png)
![image](https://user-images.githubusercontent.com/35359451/135954245-9df90cc5-e45c-4733-9e6f-1b6eb52f1bd2.png)
![image](https://user-images.githubusercontent.com/35359451/135954270-dc48d6d9-6c97-4bfe-a21b-957b59c080fb.png)
![image](https://user-images.githubusercontent.com/35359451/135954297-9e28e766-e1be-4f53-9b2d-dc78cc8df08a.png)
![image](https://user-images.githubusercontent.com/35359451/135954314-b10ab810-4d4e-41db-8805-f135af192596.png)
![image](https://user-images.githubusercontent.com/35359451/135954337-e60d9f47-5fee-411e-8e73-72ecc37e2415.png)
![image](https://user-images.githubusercontent.com/35359451/135954368-85df5b40-a991-45d1-a43d-180bc47adf65.png)
![image](https://user-images.githubusercontent.com/35359451/135954389-02c6d6d5-a7c3-4728-83c1-bda8a9d5baab.png)
![image](https://user-images.githubusercontent.com/35359451/135954415-825af5ec-b9cc-4729-b602-ccc456d88244.png)
![image](https://user-images.githubusercontent.com/35359451/135954442-a80364b7-652c-42ee-937e-ed69ee52960d.png)
![image](https://user-images.githubusercontent.com/35359451/135954471-0ef43329-7b54-456a-823a-fd8c92d4a972.png)
![image](https://user-images.githubusercontent.com/35359451/135954494-ed31cdfe-711b-4e41-b4ff-57fe79812daa.png)
