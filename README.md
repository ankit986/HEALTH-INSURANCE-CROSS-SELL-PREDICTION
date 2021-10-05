# HEALTH INSURANCE CROSS SELL PREDICTION
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
|id| (continous) |Unique identifier for the Customer.|
|Age |(continous)| Age of the Customer.|
|Gender |(dichotomous)|Gender of the Custome|
|Driving_License |(dichotomous)|0 for customer not having DL, 1 for customer having DL.|
|Region_Code |(nominal)|Unique code for the region of the customer.|
|Previously_Insured| (dichotomous)| 0 for customer not having vehicle insurance, 1 for customer having vehicle insurance.|
|Vehicle_Age| (nominal)| Age of the vehicle.|
|Vehicle_Damage | (dichotomous)| Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.|
|Annual_Premium| (continous)| The amount customer needs to pay as premium in the year.|
|Policy_Sales_Channel| (nominal)| Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.|
|Vintage |(continous)|Number of Days, Customer has been associated with the company.|
|**Response** (Dependent Feature)|(dichotomous)| 1 for Customer is interested, 0 for Customer is not interested.|


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

#### i. Decision Tree 
Decision Trees are non-parametric supervised learning methods, capable of finding complex non-linear relationships in the data. Decision trees are a type of algorithm that uses a tree-like system of conditional control statements to create the machine learning model. A decision tree observes features of an object and trains a model in the structure of a tree to predict data in the future to produce output.
For classification trees, it is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

#### ii. Gaussian Naive Bayes
Gaussian Naive Bayes is based on Bayes’ Theorem and has a strong assumption that predictors should be independent of each other. For example, Should we give a Loan applicant depending on the applicant’s income, age, previous loan, location, and transaction history? In real-life scenarios, it is most unlikely that data points don’t interact with each other but surprisingly Gaussian Naive Bayes performs well in that situation. Hence, this assumption is called class conditional independence.

#### iii. AdaBoost Classifier
Boosting is a class of ensemble machine learning algorithms that involve combining the predictions from many weak learners. A weak learner is a very simple model, although has some skill on the dataset. Boosting was a theoretical concept long before a practical algorithm could be developed, and the AdaBoost (adaptive boosting) algorithm was the first successful approach for the idea.
The AdaBoost algorithm involves using very short (one-level) decision trees as weak learners that are added sequentially to the ensemble. Each subsequent model attempts to correct the predictions made by the model before it in the sequence. This is achieved by weighing the training dataset to put more focus on training examples on which prior models made prediction errors.

#### iv. Bagging Classifier
A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

#### v. LightGBM 
Light GBM is a gradient boosting framework that uses tree-based learning algorithms. Light GBM grows trees vertically while other algorithms grow trees horizontally meaning that Light GBM grows trees leaf-wise while other algorithms grow level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, a Leaf-wise algorithm can reduce more loss than a level-wise algorithm. Light GBM is prefixed as ‘Light’ because of its high speed. Light GBM can handle the large size of data and takes lower memory to run.

#### vi. Logistic Regression
Logistic regression is named for the function used at the core of the method, the logistic function.

The logistic function, also called the sigmoid function, was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.


## 7. Hyperparameter Tuning
Tuning of hyperparameters is necessary for modeling to obtain better accuracy and to avoid overfitting. In our project, we used following techniques:

#### - GridSearchCV
#### - RandomizedSearchCV
#### - HalvingRandomizedSearchCV

![Screenshot from 2021-10-05 20-40-27](https://user-images.githubusercontent.com/35359451/136051125-b4c12fab-aa31-4908-af9e-040d14a81794.png)
![image](https://user-images.githubusercontent.com/35359451/135954494-ed31cdfe-711b-4e41-b4ff-57fe79812daa.png)


## 8. Metrics Evaluation

To evaluate our model and to obtain the accuracy and error rate of our models before and after hyperparameter tuning. We used some metric evaluation technique.
They are:
#### i. Confusion Matrix
#### ii. Accuracy
#### iii. Precision
#### iv. Recall
#### v. F1-Score
#### vi. ROC-AUC Score
#### vii. Log Loss

|||
|----|----|
| ![image](https://user-images.githubusercontent.com/35359451/135954471-0ef43329-7b54-456a-823a-fd8c92d4a972.png)|![image](https://user-images.githubusercontent.com/35359451/136068778-fc4ea607-eb68-49aa-9153-5eb91f766716.png)|
|||


**Evaluaion Before Tuning**

![image](https://user-images.githubusercontent.com/35359451/136073894-75063df3-fd0e-4e24-a8bd-29421bff8b9d.png)

**Evaluaion After Tuning**

![image](https://user-images.githubusercontent.com/35359451/136073964-35f6560b-ed77-41e0-b75f-a6dcf845aa80.png)


# Challenges Faced
- Handling Large dataset.
- Already available methods of hyper-parameter tuning were taking a huge amount of time to process.
- Memory Optimization during hyperparameter tuning.



# Conclusion
### Write your own conclusion




---
Here is a glimpse of few graphs we plotted, there are many more in the notebook please have a look.
![image](https://user-images.githubusercontent.com/35359451/136075191-a409b7a4-adf9-4950-ba79-489cb91cba69.png)|![image](https://user-images.githubusercontent.com/35359451/136075288-2879d356-8b14-4d45-9ab1-9a95e892434b.png)|![image](https://user-images.githubusercontent.com/35359451/136075349-a59d1d2a-b19a-4155-a372-077d680905ed.png)|
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/35359451/136075430-09139077-575b-444c-9bbf-fe352aac1d60.png)|![image](https://user-images.githubusercontent.com/35359451/136075516-ac13173a-9a79-4477-9f4f-f24fd3a49cba.png)|![image](https://user-images.githubusercontent.com/35359451/136076533-4c3d8e4b-ce04-4d9c-942c-38ad1b31d006.png)|
![image](https://user-images.githubusercontent.com/35359451/136076076-b7616323-5020-4646-b4a9-6e12d149fda8.png)|![image](https://user-images.githubusercontent.com/35359451/136076150-7e78971f-e134-401e-8b97-97aae61cfa46.png)|![image](https://user-images.githubusercontent.com/35359451/136076259-ffef7f15-8b6b-4230-a1c5-8e697ce06c50.png)|



---
# References
 - https://towardsdatascience.com
 - https://www.analyticsvidhya.com
 - https://machinelearningmastery.com


---
---

Meet The Team:

> Bhwesh Gaur: https://github.com/bhweshgaur

> S Sravya Sri : https://github.com/SSravyaSri

> Priyanka Dodeja :  https://github.com/PriyankaDodeja

> Ankit Bansal : https://github.com/ankit986

