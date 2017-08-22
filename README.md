Intro to Machine Learning
==================


## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 R and R Studio](#01-r-and-r-studio)
+ [0.2 Virtual Environment](#02-virtual-environment)
- [1.0 Introduction](#10-introduction)
+ [1.1 What is Machine Learning?](#11-what-is-machine-learning)
+ [1.2 Hypothesis](#12-hypothesis)
+ [1.3 Distribution](#13-distribution)
+ [1.4 Notation](#14-notation)
- [2.0 Data](#20-data)
+ [2.1 Labeled vs Unlabeled Data](#21-labeled-vs-unlabeled-data)
+ [2.2 Training vs Test Data](#22-training-vs-test-data)
+ [2.3 Overfitting vs Underfitting](#overfitting-vs-underfitting)
+ [2.4 Open Data](#24-open-data)
- [3.0 Types of Learning](#30-types-of-learning)
+ [3.1 Supervised Learning](#31-supervised-learning)
+ [3.2 Unsupervised Learning](#32-unsupervised-learning)
+ [3.3 Reinforcement Learning](#33-reinforcement-learning)
+ [3.4 Subfields](#34-subfields)
- [4.0 Fundamentals](#40-fundamentals)
+ [4.1 Inputs vs Features](#41-inputs-vs-features)
+ [4.2 Outputs vs Targets](#42-outputs-vs-targets)
+ [4.3 Function Estimation](#43-function-estimation)
* [4.3.1 Exploratory Analysis](#441-exploratory-analysis)
* [4.3.2 Data Visualization](#432-data-visualization)
* [4.3.3 Linear Model](#433-linear-model)
+ [4.4 Bias and Variance](#44-bias-and-variance)
* [4.4.1 Bias](#441-bias)
* [4.4.2 Variance](#442-variance)
* [4.4.3 Bias-Variance Tradeoff](#443-bias-variance-tradeoff)
- [5.0 Optimization](#50-optimization)
+ [5.1 Loss Function](#51-loss-function)
+ [5.2 Boosting](#52-boosting)
* [5.2.1 What is a weak learner?](#521-what-is-a-weak-learner)
* [5.2.2 What is an ensemble?](#522-what-is-an-ensemble)
+ [5.3 Occam's Razor](#53-occams-razor)
- [6.0 Final Words](#60-final-words)
+ [6.1 Resources](#61-resources)
+ [6.2 Mini Courses](#62-mini-courses)


## 0.0 Setup

This guide was written in R 3.2.3.

### 0.1 R and R Studio

Install [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

```
install.packages("ggplot2")
install.packages("caret")
```

## 1.0 Introduction

### 1.1 What is Machine Learning?

Machine Learning is the field where statistics and computer science overlap for prediction insights on data. What do we mean by <i>prediction</i>? Given a dataset, we generate an algorithm to <i>learn</i> what features or attributes are indicators of a certain label or prediction. These attributes or features can be anything that describes a data point, whether that's height, frequency, class, etc. 

This algorithm is chosen based on the original dataset, which you can think of as prior or historical data. When we refer to machine learning algorithms, we're referring to a function that best maps a data point's features to its label. A large part of the machine learning is spent improving this function as much as possible. 

### 1.2 Hypothesis

You've likely heard of hypotheses in the context of science before; typically, it's an educated guess on what and outcome will be. In the context of machine learning, a hypothesis is the function we believe is similar to the <b>true</b> function of learning - the target function that we want to model and use as our machine learning algorithm. 

### 1.3 Assumptions 

In this course, we'll review different machine learning algorithms, from decision trees to support vector machines. Each is different in its methodology and results. A critical part of the process of choosing the best algorithm is considering the assumptions you can make about the particular dataset you're working with. These assumptions can include the linearity or lack of linearity, the distribution of the dataset, and much more. 

### 1.4 Notation

In this course and future courses, you'll see lots of notation. I've collected a list of all the general notation you'll see:

<i>a</i>: scalar <br>
<i><b>a</b></i>: vector <br>
<i>A</i>: matrix <br>
<i>a<sub>i</sub></i>: the ith entry of <i><b>a</b></i> <br>
<i>a<sub>ij</sub></i>: the entry (i,j) of <i>A</i> <br>
<i><b>a<sup>(n)</sup></b></i>: the nth vector <i><b>a</b></i> in a dataset <br>
<i>A<sup>(n)</sup></i>: the nth matrix <i>A</i> in a dataset <br>
<i>H</i>: Hypothesis Space, the set of all possible hypotheses <br>


## 2.0 Data 

As a data scientist, knowing the different forms data takes is highly important. When working on a prediction problem, the collection of data you're working with is called a <i>dataset</i>.

### 2.1 Labeled vs Unlabeled Data

Generally speaking, there are two forms of data: unlabeled and labeled. Labeled data refers to data that has inputs and attached, known outputs. Unlabeled means all you have is inputs. They're denoted as follows:

- Labeled Dataset: X = {x<sup>(n)</sup> &isin; R<sup>d</sup>}<sup>N</sup><sub>n=1</sub>, Y = {y<sup>n</sup> &isin; R}<sup>N</sup><sub>n=1</sub>

- Unlabed Dataset: X = {x<sup>(n)</sup> &isin; R<sup>d</sup>}<sup>N</sup><sub>n=1</sub>

Here, X denotes a feature set containing N samples. Each of these samples is a d-dimension vector, <b>x<sup>(n)</sup></b>. Each of these dimensions is an attribute, feature, variable, or element. Meanwhile, Y is the label set. 

### 2.2 Training vs Test Data

When it comes time to train your classifier or model, you're going to need to split your data into <b>testing</b> and <b>training</b> data. 

Typically, the majority of your data will go towards your training data, while only 10-25% of your data will go towards testing. 

It's important to note there is no overlap between the two. Should you have overlap or use all your training data for testing, your accuracy results will be wrong. Any classifier that's tested on the data it's training is obviously going to do very well since it will have observed those results before, so the accuracy will be high, but wrongly so. 

Furthermore, both of these sets of data must originate from the same source. If they don't, we can't expect that a model built for one will work for the other. We handle this by requiring the training and testing data to be <b>identically and independently distributed (iid)</b>. This means that the testing data show the same distribution as the training data, but again, must not overlap.

Together these two aspects of the data are known as <i>IID assumption</i>.

### 2.3 Overfitting vs Underfitting

The concept of overfitting refers to creating a model that doesn't generaliz e to your model. In other words, if your model overfits your data, that means it's learned your data <i>too</i> much - it's essentially memorized it. 

This might not seem like it would be a problem at first, but a model that's just "memorized" your data is one that's going to perform poorly on new, unobserved data. 

Underfitting, on the other hand, is when your model is <i>too</i> generalized to your data. This model will also perform poorly on new unobserved data. This usually means we should increase the number of considered features, which will expand the hypothesis space. 

### 2.4 Open Data 

What's open data, you ask? Simple, it's data that's freely  for anyone to use! Some examples include things you might have already heard of, like APIs, online zip files, or by scraping data!

You might be wondering where this data comes from - well, it can come from a variety of sources, but some common ones include large tech companies like Facebook, Google, Instagram. Others include large institutions, like the US government! Otherwise, you can find tons of data from all sorts of organizations and individuals. 


## 3.0 Types of Learning

Generally speaking, Machine Learning can be split into three types of learning: supervised, unsupervised, and reinforcement learning. 

### 3.1 Supervised Learning

This algorithm consists of a target / outcome variable (or dependent variable) which is to be predicted from a given set of predictors (independent variables). Using these set of variables, we generate a function that map inputs to desired outputs. The training process continues until the model achieves a desired level of accuracy on the training data. Examples of Supervised Learning: Regression, Decision Tree, Random Forest, Logistic Regression, etc.

All supervised learning algorithms in the Python module scikit-learn hace a `fit(X, y)` method to fit the model and a `predict(X)` method that, given unlabeled observations X, returns predicts the corresponding labels y.

### 3.2 Unsupervised Learning

In this algorithm, we do not have any target or outcome variable to predict / estimate.  We can derive structure from data where we don't necessarily know the effect of the variables. Examples of Unsupervised Learning: Clustering, Apriori algorithm, K-means.

### 3.3 Reinforcement Learning

Using this algorithm, the machine is trained to make specific decisions. It works this way: the machine is exposed to an environment where it trains itself continually using trial and error. This machine learns from past experience and tries to capture the best possible knowledge to make accurate business decisions. Example of Reinforcement Learning: Markov Decision Process.

### 3.4 Subfields

Though Machine Learning is considered the overarching field of prediction analysis, it's important to know the distinction between its different subfields.

#### 3.4.1 Natural Language Processing

Natural Language Processing, or NLP, is an area of machine learning that focuses on developing techniques to produce machine-driven analyses of textual data.

#### 3.4.2 Computer Vision

Computer Vision is an area of machine learning and artificial intelligence that focuses on the analysis of data involving images.

#### 3.4.3 Deep Learning

Deep Learning is a branch of machine learning that involves pattern recognition on unlabeled or unstructured data. It uses a model of computing inspired by the structure of the brain, which we call this model a neural network.


## 4.0 Fundamentals

### 4.1 Inputs vs Features

The variables we use as inputs to our machine learning algorithms are commonly called inputs, but they are also frequently called predictors or features. Inputs/predictors are independent variables and in a simple 2D space, you can think of an input as x-axis variable.

### 4.2 Outputs vs Targets

The variable that we’re trying to predict is commonly called a target variable, but they are also called output variables or response variables. You can think of the target as the dependent variable; visually, in a 2-dimensional graph, the target variable is the variable on the y-axis.

#### 4.2.1 Label Types

Another name for outputs or targets is **labels**. In this section we discuss the different types of labels a dataset can encompass. 

1. Single column with binary values -- usually a classification problem where a sample belongs to one class only and there are only two classes. 
2. Single column with real values -- usually a regression problem with a prediction of only one value.
3. Multiple columns with binary values -- also usually a classification problem, where a sample belongs to one class, but there are more than two classes. 
4. Multiple columns with real values -- also usually a regression problem with a prediction of multiple values.
5. Multilabel - also a classification problem, but a sample can belong to several classes.

### 4.3 Function Estimation

When you’re doing machine learning, specifically supervised learning, you’re using computational techniques to reverse engineer the underlying function your data. 

With that said, we'll go through what this process generally looks like. In the exercise, I intentionally keep the underlying function hidden from you because we never know the underlying function in practice. Once again, machine learning provides us with a set of statistical tools for estimate <i>f(x)</i>.

So we begin with getting our data:


``` R
df.unknown_fxn_data <- read.csv(url("http://www.sharpsightlabs.com/wp-content/uploads/2016/04/unknown_fxn_data.txt"))
```

#### 4.3.1 Exploratory Analysis

Now we’ll perform some basic data exploration. First, we’ll use `str()` to examine the “structure” of the data:

``` R
str(df.unknown_fxn_data)
```

Which gets us two variables, `input_var` and `target_var`. 

``` bash
'data.frame':	28 obs. of  2 variables:
$ input_var : num  -2.75 -2.55 -2.35 -2.15 -1.95 -1.75 -1.55 -1.35 -1.15 -0.95 ...
$ target_var: num  -0.224 -0.832 -0.543 -0.725 -1.026 ...
```

Next, let’s print out the first few rows of the dataset using the head() function.

``` R
head(df.unknown_fxn_data)
```

You can think of the two columns as (x,y) pairs.

#### 4.3.2 Data Visualization

Now, let’s examine the data visually. Since `input_var` is an independent variable, we can plot it on the x-axis and `target_var` on the y-axis. And since these are (x, y) pairs, we'll use a scatterplot to visualize it.

``` R
require(ggplot2)
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
geom_point()
```

This is important because by plotting the data visually, we can visually detect a pattern which we can later use to compare to our estimated function.

#### 4.3.3 Linear Model

Now that we’ve explored our data, we’ll create a very simple linear model using caret. Here, the `train()` function is the “core” function of the caret package, which builds the machine learning model. 

Inside of the `train()` function, we’re using `df.unknown_fxn_data` for the data parameter. `target_var ~ input_var` specifies the “formula” of our model and indicates the target variable that we’re trying to predict as well as the predictor we’ll use as an input. The `method = "lm"` indicates that we want a linear regression model. 

``` R
require(caret)

mod.lm <- train( target_var ~ input_var
,data = df.unknown_fxn_data
,method = "lm"
)
```

Now that we’ve built the simple linear model using the `train()` function, let’s plot the data and plot the linear model on top of it.

To do this, we’ll extract the slope and the intercept of our linear model using the `coef()` function.

``` R
coef.icept <- coef(mod.lm$finalModel)[1]
coef.slope <- coef(mod.lm$finalModel)[2]
```

The following plots the data:

``` R
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
geom_point() +
geom_abline(intercept = coef.icept, slope = coef.slope, color = "red")
```

As I mentioned in the beginning of this example, I kept the underlying function hidden, which was a sine function with some noise. When we began this exercise, there was a hidden function that generated the data and we used machine learning to estimate that function.

``` R
set.seed(9877)
input_var <- seq(-2.75,2.75, by = .2)
target_var <- sin(input_var) + rnorm(length(input_var), mean = 0, sd = .2)
df.unknown_fxn_data <- data.frame(input_var, target_var)
```

Here, we just visualize that underlying function. 

``` R
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
geom_point() +
stat_function(fun = sin, color = "navy", size = 1)
```

### 4.4 Bias and Variance

Understanding how different sources of error lead to bias and variance helps us improve the data fitting process resulting in more accurate models.

#### 4.4.1 Bias

Error due to bias is the difference between the expected (or average) prediction of our model and the actual value we're trying to predict. 

#### 4.4.2 Variance

Error due to variance is taken as the variability of a model prediction for a given data point. In other words, the variance is how much the predictions for a given point vary between different realizations of the model.

![alt text](https://github.com/lesley2958/intro-ml/blob/master/biasvar.png?raw=true "Logo Title Text 1")

#### 4.4.3 Bias-Variance Tradeoff

The bias–variance tradeoff is the problem of simultaneously minimizing two sources of error that prevent supervised learning algorithms from generalizing beyond their training set.

## 5.0 Optimization

In the simplest case, optimization consists of maximizing or minimizing a function by systematically choosing input values from within an allowed set and computing the value of the function. 

### 5.1 Loss Function

The job of the loss function is to tell us how inaccurate a machine learning system's prediction is in comparison to the truth. It's denoted with &#8467;(y, y&#770;), where y is the true value and y&#770; is a the machine learning system's prediction. 

The loss function specifics depends on the type of machine learning algorithm. In Regression, it's (y - y&#770;)<sup>2</sup>, known as the squared loss. Note that the loss function is something that you must decide on based on the goals of learning. 

Since the loss function gives us a sense of the error in a machine learning system, the goal is to <i>minimize</i> this function. Since we don't know what the distribution does, we have to calculte the loss function for each data point in the training set, so it becomes: 

![alt text](https://github.com/lesley2958/intro-ml/blob/master/trainingerror.png?raw=true "Logo Title Text 1")

In other words, our training error is simply our average error over the training data. Again, as we stated earlier, we can minimize this to 0 by memorizing the data, but we still want it to generalize well so we have to keep this in mind when minimizing the loss function. 

### 5.2 Boosting

Boosting is a machine learning meta-algorithm that iteratively builds an ensemble of weak learners to generate a strong overall model.

#### 5.2.1 What is a weak learner?

A <i>weak learner</i> is any machine learning algorithm that has an accuracy slightly better than random guessing. For example, in binary classification, approximately 50% of the samples belong to each class, so a weak learner would be any algorithm that slightly improves this score – so 51% or more. 

These weak learners are usually fairly simple because using complex models usually leads to overfitting. The total number of weak learners also needs to be controlled because having too few will cause underfitting and have too many can also cause overfitting.

#### 5.2.2 What is an ensemble?

The overall model built by Boosting is a weighted sum of all of the weak learners. The weights and training given to each ensures that the overall model yields a pretty high accuracy.

#### 5.2.3 What do we mean by iteratively build?

Many ensemble methods train their components in parallel because the training of each those weak learners is independent of the training of the others, but this isn't the case with Boosting. 

At each step, Boosting tries to evaluate the shortcomings of the overall model built, and then generates a weak learner to battle these shortcomings. This weak learner is then added to the total model. Therefore, the training must necessarily proceed in a serial/iterative manner.

Each of the iterations is basically trying to improve the current model by introducing another learner into the ensemble. Having this kind of ensemble reduces the bias and the variance. 

#### 5.2.4 Gradient Boosting


#### 5.2.5 AdaBoost


#### 5.2.6 Disadvantages

Because boosting involves performing so many iterations and generating a new model at each, a lot of computations, time, and space are utilized.

Boosting is also incredibly sensitive to noisy data. Because boosting tries to improve the output for data points that haven't been predicted well. If the dataset has misclassified or outlier points, then the boosting algorithm will try to fit the weak learners to these noisy samples, leading to overfitting. 

### 5.3 Occam's Razor

Occam's Razor states that any phenomenon should make as few assumptions as possible. Said again, given a set of possible solutions, the one with the fewest assumptions should be selected. The problem here is that Machine Learning often puts accuracy and simplicity in conflict.

## 6.0 Final Words

What we've covered here is a very small glimpse of machine learning. Applying these concepts in actual machine learning algorithms such as random forests, regression, classification, is the harder but doable part. 

### 6.1 Resources

[Stanford Coursera](https://www.coursera.org/learn/machine-learning) <br>
[Math &cap; Programming](https://jeremykun.com/main-content/)

========================================================================


# Machine Learning with Naive Bayes


## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 Python and Pip](#01-python-and-pip)
- [1.0 Introduction](#10-introduction)
+ [1.1 Bayes Theorem](#11-bayes-theorem)
+ [1.2 Overview](#12-overview)
* [1.2.1 Frequency Table](#121-frequency-table)
* [1.2.2 Likelihood Table](#122-likelihood-table)
* [1.2.3 Calculation](#123-calculation)
+ [1.3 Naive Bayes Evaluation](#13-naive-bayes-evaluation)
* [1.3.1 Pros](#131-pros)
* [1.3.2 Cons](#132-cons)
- [2.0 Naive Bayes Types](#20-naive-bayes-types)
+ [2.1 Gaussian](#21-gaussian)
+ [2.2 Multinomial](#22-multinomial)
+ [2.3 Bernoulli](#23-bernoulli)
+ [2.4 Tips for Improvement](#24-tips-for-improvement)
- [3.0 Sentiment Analysis](#30-sentiment-analysis)
+ [3.1 Preparing the Data](#31-preparing-the-data)
* [3.1.1 Training Data](#311-training-data)
* [3.1.2 Test Data](#312-test-data)
+ [3.2 Building a Classifier](#32-building-a-classifier)
+ [3.3 Classification](#33-classification)
+ [3.4 Accuracy](#34-accuracy)


## Introduction

In this tutorial set, we'll review the Naive Bayes Algorithm used in the field of machine learning. Naive Bayes works on Bayes Theorem of probability to predict the class of a given data point, and is extremely fast compared to other classification algorithms. 

Because it works with an assumption of independence among predictors, the Naive Bayes model is easy to build and particularly useful for large datasets. Along with its simplicity, Naive Bayes is known to outperform even some of the most sophisticated classification methods.

This tutorial assumes you have prior programming experience in Python and probablility. While I will overview some of the priciples in probability, this tutorial is **not** intended to teach you these fundamental concepts. If you need some background on this material, please see my tutorial [here](https://github.com/lesley2958/intro-stats).


### Bayes Theorem

Recall Bayes Theorem, which provides a way of calculating the *posterior probability*: 

![alt text](https://github.com/lesley2958/intro-stats/blob/master/images/bayes.png?raw=true)

Before we go into more specifics of the Naive Bayes Algorithm, we'll go through an example of classification to determine whether a sports team will play or not based on the weather. 

To start, we'll load in the data, which you can find [here](https://github.com/lesley2958/ml-bayes/blob/master/data/weather.csv).

### 1.3 Naive Bayes Evaluation

Every classifier has pros and cons, whether that be in terms of computational power, accuracy, etc. In this section, we'll review the pros and cons of Naive Bayes.

#### 1.3.1 Pros

Naive Bayes is incredibly easy and fast in predicting the class of test data. It also performs well in multi-class prediction.

When the assumption of independence is true, the Naive Bayes classifier performs better thanother models like logistic regression. It does this, and with less need of a lot of data.

Naive Bayes also performs well with categorical input variables compared to numerical variable(s), which is why we're able to use it for text classification. For numerical variables, normal distribution must be assumed.

#### 1.3.2 Cons

If a categorical variable has a category not observed in the training data set, then model will assign a 0 probability and will be unable to make a prediction. This is referred to as “Zero Frequency”. To solve this, we can use the smoothing technique, such as Laplace estimation.

Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.


## 2.0 Naive Bayes Types

With `scikit-learn`, we can implement Naive Bayes models in Python. There are three types of Naive Bayes models, all of which we'll review in the following sections.

### 2.1 Gaussian

The Gaussian Naive Bayes Model is used in classification and assumes that features will follow a normal distribution. 

We begin an example by importing the needed modules:


### 2.2 Multinomial

MultinomialNB implements the multinomial Naive Bayes algorithm and is one of the two classic Naive Bayes variants used in text classification. This classifier is suitable for classification with discrete features (such as word counts for text classification). 

The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts may also work.

### 2.3 Bernoulli

Like MultinomialNB, this classifier is suitable for discrete data. BernoulliNB implements the Naive Bayes training and classification algorithms for data that is distributed according to multivariate Bernoulli distributions, meaning there may be multiple features but each one is assumed to be a binary value. 

The decision rule for Bernoulli Naive Bayes is based on

![alt text](https://github.com/lesley2958/ml-bayes/blob/master/bernoulli.png?raw=true "Logo Title Text 1")

### 2.4 Tips for Improvement

If continuous features don't have a normal distribution (recall the assumption of normal distribution), you can use different methods to convert it to a normal distribution.

As we mentioned before, if the test data set has a zero frequency issue, you can apply smoothing techniques “Laplace Correction” to predict the class.

As usual, you can remove correlated features since the correlated features would be voted twice in the model and it can lead to over inflating importance.


## 3.0 Sentiment Analysis  

So you might be asking, what exactly is "sentiment analysis"? 

Well, sentiment analysis involves building a system to collect and determine the emotional tone behind words. This is important because it allows you to gain an understanding of the attitudes, opinions and emotions of the people in your data. 

At a high level, sentiment analysis involves Natural language processing and artificial intelligence by taking the actual text element, transforming it into a format that a machine can read, and using statistics to determine the actual sentiment.

### 3.1 Preparing the Data 

To accomplish sentiment analysis computationally, we have to use techniques that will allow us to learn from data that's already been labeled. 

So what's the first step? Formatting the data so that we can actually apply NLP techniques.

## 4.0 Joint Models

If you have input data x and want to classify the data into labels y. A <b>generative model</b> learns the joint probability distribution `p(x,y)` and a discriminative model learns the <b>conditional probability distribution</b> `p(y|x)`.

Here's an simple example of this form:

```
(1,0), (1,0), (2,0), (2, 1)
```

`p(x,y)` is

```
y=0   y=1
-----------
x=1 | 1/2   0
x=2 | 1/4   1/4
```

Meanwhile,

```
p(y|x) is
```

```
y=0   y=1
-----------
x=1 | 1     0
x=2 | 1/2   1/2
```

Notice that if you add all 4 probabilities in the first chart, they add up to 1, but if you do the same for the second chart, they add up to 2. This is because the probabilities in chart 2 are read row by row. Hence, `1+0=1` in the first row and `1/2+1/2=1` in the second.

The distribution `p(y|x)` is the natural distribution for classifying a given example `x` into a class `y`, which is why algorithms that model this directly are called discriminative algorithms. 

Generative algorithms model `p(x, y)`, which can be tranformed into `p(y|x)` by applying Bayes rule and then used for classification. However, the distribution `p(x, y)` can also be used for other purposes. For example you could use `p(x,y)` to generate likely `(x, y)` pairs.


==========================================================================================


# Machine Learning with Classification

## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 R and R Studio](#01-r-and-r-studio)
+ [0.2 Python and Pip](#02-python-and-pip)
+ [0.3 Virtual Environment](#03-virtual-environment)
- [1.0 Introduction](#10-introduction)
- [2.0 Nearest Neighbor](#20-nearest-neighbor)
+ [2.1 k-Nearest Neighbor Classifier](#21-k-nearest-neighbor-classifier)
- [3.0 Support Vector Machines](#30-support-vector-machines)
+ [3.1 Separating Hyperplane](#31-separating-hyperlanes)
* [3.1.1 Optimal Separating Hyperplane](#311-optimal-separating-hyperplane)
+ [3.2 Margins](#32-margins)
+ [3.3 Equation](#33-equations)
* [3.3.1 Example](#331-example)
* [3.3.2 Margin Computation](#332-margin-computation)
- [4.0 Iris Classification](#40-iris-classification)
+ [4.1 Kernels](#41-kernels)
+ [4.2 Tuning](#42-tuning)
* [4.2.1 Gamma](#421-gamma)
* [4.2.2 C value](#422-c-value)
+ [4.3 Evaluation](#43-evaluation)
* [4.3.1 Pros](#431-pros)
* [4.3.2 Cons](#432-cons)


## 0.0 Setup

This guide was written in R 3.2.3.

### 0.1 R and R Studio

Install [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

```
install.packages("RTextTools")
```
## 1.0 Introduction

Classification is a machine learning algorithm whose output is within a set of discrete labels. This is in contrast to Regression, which involved a target variable that was on a continuous range of values.

Throughout this tutorial we'll review different classification algorithms, but you'll notice that the workflow is consistent: split the dataset into training and testing datasets, fit the model on the training set, and classify each data point in the testing set. 


## 2.0 LDA vs QDA

Discriminant Analysis is a statistical technique used to classify data into groups based on each data point's features. 

``` R
library(dplyr)
library(magrittr)
olives <- readRDS("olives.RDS")

split <- createDataPartition(y=olives$Region,
p=0.80,
list=FALSE)
train <- olives[split,]
test <- olives[-split,]
```

In R, there's a library called `MASS` that provides us with built-in functions to perform discriminant analysis. 

``` R
library(MASS)
```

### 2.1 Linear Discriminant Analysis

Linear Discriminant Analysis (LDA) is a linear classification algorithm used for when it can be assumed that the covariance is the same for <b>all</b> classes. Its mostly used as a dimension reduction technique in the pre-processing portion so that the different classes are separable and therefore easier to classify and avoid overfitting. 

LDA is similar to Principal Component Analysis (PCA), but LDA also considers the axes that maximize the separation between multiple classes. It works by finding the linear combinations of the original variables that gives the best separation between the groups in the data set. 

``` R
lda <- lda(train$Region ~ ., data=train[,3:10])
plot(lda)
```

``` R
train_predict <- predict(lda, train[,3:10])
(t <- table(train$Region, train_predict$class))
test_predict <- predict(lda, test[,3:10])
(t <- table(test$Region, test_predict$class))
```

### 2.2 Quadratic Discriminant Analysis

Quadratic Discriminant Analysis, on the other hand, is used for heterogeneous variance-covariance matrices. Because QDA has more parameters to estimate, it's typically less accurate than LDA. 

``` R
qda <- qda(train$Region ~ ., data=train[,3:10])
```

``` R
train_predict <- predict(qda, train[,3:10])
(t <- table(train$Region, train_predict$class))
test_predict <- predict(qda, test[,3:10])
(t <- table(test$Region, test_predict$class))
```
## 3.0 Support Vector Machines

Support Vector Machines (SVMs) is a machine learning algorithm used for classification tasks. Its primary goal is to find an optimal separating hyperplane that separates the data into its classes. This optimal separating hyperplane is the result of the maximization of the margin of the training data.

### 3.1 Separating Hyperplane

Let's take a look at the scatterplot of the iris dataset that could easily be used by a SVM: 

![alt text](https://github.com/lesley2958/regression/blob/master/log-scatter.png?raw=true "Logo Title Text 1")

Just by looking at it, it's fairly obvious how the two classes can be easily separated. The line which separates the two classes is called the <b>separating hyperplane</b>.

In this example, the hyperplane is just two-dimensional, but SVMs can work in any number of dimensions, which is why we refer to it as <i>hyperplane</i>.

#### 3.1.1 Optimal Separating Hyperplane

Going off the scatter plot above, there are a number of separating hyperplanes. The job of the SVM is find the <i>optimal</i> one. 

To accomplish this, we choose the separating hyperplane that maximizes the distance from the datapoints in each category. This is so we have a hyperplane that generalizes well.

### 3.2 Margins

Given a hyperplane, we can compute the distance between the hyperplane and the closest data point. With this value, we can double the value to get the <b>margin</b>. Inside the margin, there are no datapoints. 

The larger the margin, the greater the distance between the hyperplane and datapoint, which means we need to <b>maximize</b> the margin. 

![alt text](https://github.com/lesley2958/ml-classification/blob/master/margin.png?raw=true "Logo Title Text 1")

### 3.3 Equation

Recall the equation of a hyperplane: w<sup>T</sup>x = 0. Here, `w` and `x` are vectors. If we combine this equation with `y = ax + b`, we get:

![alt text](https://github.com/lesley2958/ml-classification/blob/master/wt.png?raw=true "Logo Title Text 1")

This is because we can rewrite `y - ax - b = 0`. This then becomes:

w<sup>T</sup>x = -b &Chi; (1) + (-a) &Chi; x + 1 &Chi; y

This is just another way of writing: w<sup>T</sup>x = y - ax - b. We use this equation instead of the traditional `y = ax + b` because it's easier to use when we have more than 2 dimensions.  

#### 3.3.1 Example

Let's take a look at an example scatter plot with the hyperlane graphed: 

![alt text](https://github.com/lesley2958/ml-classification/blob/master/ex1.png?raw=true "Logo Title Text 1")

Here, the hyperplane is x<sub>2</sub> = -2x<sub>1</sub>. Let's turn this into the vector equivalent:

![alt text](https://github.com/lesley2958/ml-classification/blob/master/vectex1.png?raw=true "Logo Title Text 1")

Let's calculate the distance between point A and the hyperplane. We begin this process by projecting point A onto the hyperplane.

![alt text](https://github.com/lesley2958/ml-classification/blob/master/projex1.png?raw=true "Logo Title Text 1")

Point A is a vector from the origin to A. So if we project it onto the normal vector w: 

![alt text](https://github.com/lesley2958/ml-classification/blob/master/normex1.png?raw=true "Logo Title Text 1")

This will get us the projected vector! With the points (3,4) and (2,1) [this came from w = (2,1)], we can compute `||p||`. Now, it'll take a few steps before we get there.

We begin by computing `||w||`: 

`||w||` = &#8730;(2<sup>2</sup> + 1<sup>2</sup>) = &#8730;5. If we divide the coordinates by the magnitude of `||w||`, we can get the direction of w. This makes the vector u = (2/&#8730;5, 1/&#8730;5).

Now, p is the orthogonal prhoojection of a onto w, so:

![alt text](https://github.com/lesley2958/ml-classification/blob/master/orthproj.png?raw=true "Logo Title Text 1")

#### 3.3.2 Margin Computation

Now that we have `||p||`, the distance between A and the hyperplane, the margin is defined by:

margin = 2||p|| = 4&#8730;5. This is the margin of the hyperplane!


## 4.0 Iris Classification

Let's perform the iris classification from earlier with a support vector machine model:

### 4.1 Kernels

Classes aren't always easily separable. In these cases, we build a decision function that's polynomial instead of linear. This is done using the <b>kernel trick</b>.

![alt text](https://github.com/lesley2958/regression/blob/master/svm.png?raw=true "Logo Title Text 1")

If you modify the previous code with the kernel parameter, you'll get different results!

### 4.3 Evaluation

As with any other machine learning model, there are pros and cons. In this section, we'll review the pros and cons of this particular model.

#### 4.3.1 Pros

SVMs largest stength is its effectiveness in high dimensional spaces, even when the number of dimensions is greater than the number of samples. Lastly, it uses a subset of training points in the decision function (called support vectors), so it's also memory efficient.

#### 4.3.2 Cons

Now, if we have a large dataset, the required time becomes high. SVMs also perform poorly when there's noise in the dataset. Lastly, SVM doesn’t directly provide probability estimates and must be calculated using an expensive five-fold cross-validation.


========================================================================


# Unsupervised Machine Learning


## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 Python and Pip](#01-python-and-pip)
+ [0.2 R and R Studio](#02-r-and-r-studio)
+ [0.3 Other](#03-other)
+ [0.4 Virtual Environment](#04-virtual-environment)
- [1.0 Introduction](#10-introduction)
- [6.0 Final Words](#50-final-words)
+ [6.2 Mini Courses](#62-mini-courses)

## 0.0 Setup

## 1.0 Introduction

As we've covered before, there are two general categories that machine learning falls into. First is supervised learning, which we've covered with regression analysis, decision trees, and support vector machines. 

Recall that supervised learning is when your explanatory variables X come with an target variable Y. In contrast, unsupervised learning has no labels, so we a lot of X's with no Y's. In unsupervised learning all we can do is try our best to extract some meaning out of the data's underlying structure and do some checks to make sure that our methods are robust.

### 1.1 Clustering 

One example of an unsupervised learning algorithm is clustering! Clustering is exactly what it sounds like. It's a way of grouping “similar” data points together into clusters or subgroups, while keeping each group as distinct as possible. 

In this way data points belonging to different clusters will be quite different from each other, too. This is useful because oftentimes we'll come across datasets which exhibit this kind of grouped structure. Now, you might be thinking how are two points considered similar? That's a fair point and there are two ways in which we determine that: 1. Similarity 2. Cluster centroid. We'll go into detail on what these two things mean in the next section. 

### 1.2 Similarity 

Intuitively, it makes sense that similar things should be close to each other, while different things should be farther apart. So to formalize the notion of similarity, we choose a distance metric (see below) that can quantify exactly how "close" two points are to each other. The most commonly used distance metric is the Euclidean distance which we should all be pretty familiar with (think: distance formula from middle school), and that's what we'll be using in our example today. We'll introduce some other distance metrics towards the end.

### 1.3 Cluster Centroid

The cluster centroid is the most representative feature of the entire cluster. We say "feature" instead of "point" because the centroid may not necessarily be an existing point in the cluster. You can find it by averaging the values of all the points belonging to a specific group. But any relevant information about the cluster centroid tells us everything that we need to know about all other points in the same cluster.


## 2.0 K Means Clustering

The k-means algorithm has a simple objective: given a set of data points, it tries to separate them out into k distinct clusters. It uses the same principle that we mentioned earlier: keep the data points within each cluster as similar as possible. You have to provide the value of k to the algorithm, so you should have a general idea of how many clusters you're expecting to see in your data. This sin't a precise science, but we can utilize visualization techniques to help us choose a proper k. 

So let’s begin by doing just that. Remember that clustering is an unsupervised learning method, so we’re never going to have a perfect answer for our final clusters, but we'll do our best to make sure that the results we get are reasonable and replicable. 

By replicable, we mean that our results can be arrived at by someone else using a different starting point. By reasonable, we mean that our results have to show some correlation with what we expect to encounter in real life.

The following image is just an example of the visualization we might get. Notice the three colors and the ways in which they could be separated, so we can set k to 3. Right now we’re operating under the assumption that we know how many clusters we want, but we’ll go into more detail about relaxing this assumption and how to choose the best possible k at the end of the workshop.

![alt text](https://camo.githubusercontent.com/6e540cb12555953bf43925fc20d46b6da1768017/687474703a2f2f707562732e7273632e6f72672f73657276696365732f696d616765732f525343707562732e65506c6174666f726d2e536572766963652e46726565436f6e74656e742e496d616765536572766963652e7376632f496d616765536572766963652f41727469636c65696d6167652f323031322f414e2f6332616e3136313232622f6332616e3136313232622d66332e676966 "Logo Title Text 1")

### 2.1 Centroid Initialization

First we initialize three random cluster centroids. We initialize these clusters randomly because every iteration of k-means will "correct" them towards the right clusters. Since we are heading to a correct answer anyway, we don't really care about where we start.

As we explained before, these centroids are our “representative points” -- they contain all the information that we need about other points in the same cluster. It makes sense to think about these centroids as being the physical center of each cluster, so let’s pretend like our randomly initialized cluster centers are the actual centroids, and group our points accordingly. Here we use our distance metric of choice, in this case the Euclidean distance. So for every single data point we have, we compute the two distances: one from the first cluster centroid, and the other from the second centroid. We assign this data point to the cluster at which the distance to the centroid is the smallest. This makes sense, because intuitively we’re grouping points which are closer together.


### 2.2 Cluster Formation

Now we have something that’s starting to resemble three distinct clusters! But remember that we need to update the centroids that we started with -- we’ve just added in a bunch of new data points to each cluster, so we need our “representative point,” or our centroid, to reflect that.

So we’ll just do quick averaging of all the values within each cluster and call that our new centroid. The new centroids are further "within" the data than the older centroids. Notice that we’re not quite done yet -- we have some straggling points which don’t really seem to belong in either cluster. Let’s run another iteration of k-means and see if that separates out the clusters better. So recall that we’re just computing the distances from the centroids for each data point, and re-assigning those that are closer to centroids of the other cluster.


### 2.3 Iteration

We keep computing the centroids for every iteration using the steps before. After doing the few iterations, maybe you’ll notice that the clusters don’t change after a certain point. This actually turns out to be a good criterion for stopping the cluster iterations! At that point we’re just wasting time and computational resources. So let’s formalize this idea of a “stopping criterion.” We define a small value, &epsilon;, and we can terminate the algorithm when the change in cluster centroids is less than epsilon. This way, epsilon serves as a measure of how much error we can tolerate.


## 3.0 Image Segmentation

Now we'll move onto a k-means example with images! 

Images often have a few dominant colors -- for example, the bulk of the image is often made up of the foreground color and the background color. In this example, we'll write some code that uses scikit-learn's k-means clustering implementation to find the what these dominant colors may be.

Once we know what the most important colors are in an image, we can compress (or "quantize") the image by re-expressing the image using only the set of k colors that we get from the algorithm. We'll be analyzing the two following images:

![alt text](https://github.com/adicu/AccessibleML/blob/master/datasets/kmeans/imgs/leo_bb.png?raw=true "Logo Title Text 1")

![alt text](https://github.com/adicu/AccessibleML/blob/master/datasets/kmeans/imgs/mario.png?raw=true "Logo Title Text 1")

## 4.0 Limitations and Extensions

In our very first example, we started with k = 3 centroids. In case you're wondering how we arrived at this magic number and why, read on.

### 4.1 Known Number of Centroids 

Sometimes, you may be in a situation where the number of clusters is provided to you beforehand. For example, you may be asked to categorize a vast range of different bodily actions to the three main subdivisions of the brain (cerebrum, cerebellum and medulla). 

Here you know that you are looking for three main clusters where each cluster will represent the part of the brain the data point is grouped to. So in this situation, you expect to have three centroids.


### 4.2 Unknown Number of Centroids

However, there may be other situations while training in which you may not even know how many centroids to pick up from your data. Two extreme situations generally happen.

*
#### 4.2.1 Extreme Cases

You could either end up making each point its own representative (a perfect centroid) at the risk of losing any grouping tendencies. This is usually called the overfitting problem. While each point perfectly represents itself, it gives you no general information about the data as a whole and will be unable to tell you anything relevant about new data that is coming in.

You could end up choosing only one centroid from all the data (a perfect grouping). Since there is no way to generalize an enormous volume of data to one point alone, this method loses relevant distinguishing features of the data.This is kind of like saying that all the people in the world drink water, so we can cluster them all by this feature. In Machine Learning terminology, this is called the underfitting problem. Underfitting implies that we are generalizing all of our data to a potentially trivial common feature.

#### 4.2.2 Stability

Unfortunately, there’s no easy way to determine the optimal value of k. It’s a hard problem: we have to think about balancing out the number of clusters that makes the most sense for our data, while at the same time making sure that we don’t overfit our model to the exact dataset that we have. There are a few ways that we can address this, and we’ll briefly mention them here.

The most intuitive explanation is the idea of stability. If the clusters we obtain represent a true, underlying pattern in our data, it makes sense that the clusters shouldn’t change very much on separate but similar samples. So if we randomly subsample or split our data into smaller parts and run the clustering algorithm again, the cluster memberships shouldn’t drastically change. If they did, that’d be an indication that our clusters were too finely-tuned to the random noise in our data. Therefore, we can compute “stability scores” for a fixed value of k and observe which value of k gives us the most stable clusters. This idea of perturbation is really important for machine learning in general, and will come up time and time again.

We can also use penalization approaches, where we use different criterion such as AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) to keep the value of k under control.


==================================================================================

# Machine Learning & Tree Modeling


## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 Python and Pip](#01-python-and-pip)
+ [0.2 Libraries](#02-libraries)
+ [0.3 R and RStudio](#03-r-and-rstudio)
- [1.0 Background](#10-background)
- [2.0 Decision Trees](#20-decision-trees)
+ [2.1 Decision Trees in R](#21-decision-trees-in-r)
+ [2.2 Pruning Decision Trees](#22-pruning-decision-trees)
+ [2.3 Prediction](#23-prediction)
- [3.0 Random Forests](#30-random-forests)
+ [3.1 Algorithm](#31-algorithm)
+ [3.2 Advantages](#32-advantages)
+ [3.3 Limitations](#33-limitations)
+ [3.4 Random Forests in R](#34-random-forests-in-r)
+ [3.5 Variable Importance](#35-variable-importance)


## 0.0 Setup


### 0.3 R and RStudio

Install [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

```
install.packages("caret")
install.packages("partykit")
install.packages("party")
install.packages("rpart")
install.packages("randomForest")

```

## 1.0 Background

Recall in data structures learning about the different types of tree structures - binary, red black, and splay trees. In tree based modeling, we work off these structures for classification prediction. 

Tree based machine learning is great because it's incredibly accurate and stable, as well as easy to interpret. Despite being a linear model, tree based models map non-linear relationships well. The general structure is as follows: 


## 2.0 Decision Trees

Decision trees are a type of supervised learning algorithm used in classification that works for both categorical and continuous input/output variables. This typle of model includes structures with nodes which represent tests on attributes and the end nodes (leaves) of each branch represent class labels. Between these nodes are what we call edges, which represent a 'decision' that separates the data from the previous node based on some criteria. 

![alt text](https://www.analyticsvidhya.com/wp-content/uploads/2016/04/dt.png "Logo Title Text 1")

Looks familiar, right? 

### 2.1 Nodes

As mentioned above, nodes are an important part of the structure of Decision Trees. In this section, we'll review different types of nodes.

#### 2.1.1 Root Node

The root node is the node at the very top. It represents an entire population or sample because it has yet to be divided by any edges. 

#### 2.1.2 Decision Node

Decision Nodes are the nodes that occur between the root node and leaves of your decision tree. It's considered a decision node because it's a resulting node of an edge that then splits once again into either more decision nodes, or the leaves.

#### 2.1.3 Leaves/Terminal Nodes

As mentioned before, leaves are the final nodes at the bottom of the decision tree that represent a class label in classification. They're also called <i>terminal nodes</i> because more nodes do not split off of them. 

#### 2.1.4 Parent and Child Nodes

A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.

### 2.2 Pros & Cons

#### 2.2.1 Pros

1. Easy to Understand: Decision tree output is fairly easy to understand since it doesn't require any statistical knowledge to read and interpret them. Its graphical representation is very intuitive and users can easily relate their hypothesis.

2. Useful in Data exploration: Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables. With the help of decision trees, we can create new variables / features that has better power to predict target variable. You can refer article (Trick to enhance power of regression model) for one such trick.  It can also be used in data exploration stage. For example, we are working on a problem where we have information available in hundreds of variables, there decision tree will help to identify most significant variable.

3. Less data cleaning required: It requires less data cleaning compared to some other modeling techniques. It is not influenced by outliers and missing values to a fair degree.

4. Data type is not a constraint: It can handle both numerical and categorical variables.

5. Non Parametric Method: Decision tree is considered to be a non-parametric method. This means that decision trees have no assumptions about the space distribution and the classifier structure.

#### 2.2.2 Cons

1. Over fitting: Over fitting is one of the most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning (discussed in detailed below).

2. Not fit for continuous variables: While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.

### 2.1 Decision Trees in R

We begin by loading the required libraries:

``` R
library(rpart)
library(party)
library(partykit)
```

In this example, we'll be working with credit scores, so we load the needed data:

``` R
library(caret)
data(GermanCredit)
```

As always, we split the data into training and test data:

``` R
inTrain <- runif(nrow(GermanCredit)) < 0.2
```

Now, we can build the actual decision tree with `rpart()`. 

``` R
dt <- rpart(Class ~ Duration + Amount + Age, method="class",data=GermanCredit[inTrain,])
```

But let's check out the internals of this decision tree. Let's plot it using the `plot()` function: 

``` R
plot(dt)
text(dt)
```

And to prettify it, we type:

``` R
plot(as.party(dt))
```

We obviously want to have the best machine learning models for our data. To make sure we have optimal models, we can calculate the complexity parameter:

``` R
printcp(dt)
```

``` 
Classification tree:
rpart(formula = Class ~ Duration + Amount + Age, data = GermanCredit[inTrain, 
], method = "class")

Variables actually used in tree construction:
[1] Age      Amount   Duration

Root node error: 61/199 = 0.30653

n= 199 

CP nsplit rel error xerror    xstd
1 0.065574      0   1.00000 1.0000 0.10662
2 0.057377      1   0.93443 1.0984 0.10929
3 0.032787      3   0.81967 1.0656 0.10846
4 0.016393      5   0.75410 1.0656 0.10846
5 0.010000      6   0.73770 1.1967 0.11145
```

In this output, the rows show result for trees with different numbers of nodes. The column `xerror` represents the cross-validation error and the `CP` represents the complexity parameter. 

### 2.2 Pruning Decision Trees

Decision Tree pruning is a technique that reduces the size of decision trees by removing sections (nodes) of the tree that provide little power to classify instances. This is great because it reduces the complexity of the final classifier, which results in increased predictive accuracy by reducing overfitting. 

Ultimately, our aim is to reduce the cross-validation error. First, we index with the smallest complexity parameter:

``` R
m <- which.min(dt$cptable[, "xerror"])
```

To get the optimal number of splits, we:

``` R
dt$cptable[m, "nsplit"]
```

Now, we choose the corresponding complexity parameter:

``` R
dt$cptable[m, "CP"]
```

Now, we're ready for pruning:

``` R
p <- prune(dt, cp = dt$cptable[which.min(dt$cptable[, "xerror"]), "CP"])
plot(as.party(p))
```

### 2.3 Prediction

Now we can get into the prediction portion. We can use the `predict()` function in R:

``` R
pred <- predict(p, GermanCredit[-inTrain,], type="class")
```

Now let's get the confusion matrix:

``` R
table(pred=pred, true=GermanCredit[-inTrain,]$Class)
```

## 3.0 Random Forests

Recall the ensemble learning method from the Optimization lecture. Random Forests are an ensemble learning method for classification and regression. It works by combining individual decision trees through bagging. This allows us to overcome overfitting. 

### 3.1 Algorithm

First, we create many decision trees through bagging. Once completed, we inject randomness into the decision trees by allowing the trees to grow to their maximum sizes, leaving them unpruned. 

We make sure that each split is based on randomly selected subset of attributes, which reduces the correlation between different trees. 

Now we get into the random forest by voting on categories by majority. We begin by splitting the training data into K bootstrap samples by drawing samples from training data with replacement. 

Next, we estimate individual trees t<sub>i</sub> to the samples and have every regression tree predict a value for the unseen data. Lastly, we estimate those predictions with the formula:

![alt text](https://github.com/lesley2958/ml-tree-modeling/blob/master/rf-pred.png?raw=true "Logo Title Text 1")

where y&#770; is the response vector and x = [x<sub>1</sub>,...,x<sub>N</sub>]<sup>T</sup> &isin; X as the input parameters. 


### 3.2 Advantages

Random Forests allow us to learn non-linearity with a simple algorithm and good performance. It's also a fast training algorithm and resistant to overfitting.

What's also phenomenal about Random Forests is that increasing the number of trees decreases the variance without increasing the bias, so the worry of the variance-bias tradeoff isn't as present. 

The averaging portion of the algorithm also allows the real structure of the data to reveal. Lastly, the noisy signals of individual trees cancel out. 

### 3.3 Limitations 

Unfortunately, random forests have high memory consumption because of the many tree constructions. There's also little performance gain from larger training datasets. 

### 3.4 Random Forests in R

We begin by loading the required libraries and data: 

``` R
library(randomForest)
library(caret)
data(GermanCredit)
```

Once again, we split our data into training and test data.
``` R
inTrain <- runif(nrow(GermanCredit)) < 0.2
```

Now we can run the random forest algorithm on this data:

``` R
rf <- randomForest(Class ~ .,
data=GermanCredit[inTrain,],
ntree=100)
```

Let's take a look at the estimated error across the number of decision trees. The dotted lines represent the individual errors and the solid black line represents the overall error. 

``` R
library(rf)
```


Let's check out the confusion matrix:

``` R
rf$confusion
```

And finally, let's get back to predicting: 

``` R
pred <- predict(rf, newdata=GermanCredit[-inTrain,])
table(pred=pred, true=GermanCredit$Class[-inTrain])
```

### 3.5 Variable Importance


![alt text](https://github.com/lesley2958/ml-tree-modeling/blob/master/var%20importance.png?raw=true "Logo Title Text 1")

``` R
rf2 <- randomForest(Class ~ .,
data=GermanCredit, #with full dataset
ntree=100,
importance=TRUE)
```

Now, let's plot it. `type` chooses the importance metirx (1 denotes the mean decrease in accuracy if the variable were randomly permuted). `n.var` denotes the number of variables:

``` R
varImpPlot(rf2, type=1, n.var=5)
```


==================================================================================


# Machine Learning Optimization


## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 Python and Pip](#01-python-and-pip)
+ [0.2 Libraries](#02-libraries)
- [1.0 Background](#10-background)
- [2.0 Ensemble Learning](#20-ensemble-learning)
- [3.0 Bagging](#30-bagging)
+ [3.1 Algorithm](#31-algorithm)
- [4.0 Boosting](#40-boosting)
+ [4.1 Algorithm](#41-algorithm)
+ [4.2 Boosting in R](#42-boosting-in-r)
- [5.0 AdaBoosting](#50-adaboosting)
+ [5.1 Benefits](#51-benefits)
+ [5.2 Limits](#52-limits)
+ [5.3 AdaBoost in R](#53-adaboost-in-r)

## 0.0 Setup


## 1.0 Background


## 2.0 Ensemble Learning

Ensemble Learning allows us to combine predictions from different multiple learning algorithms. This is what we consider to be the "ensemble". By doing this, we can have a result with a better predictive performance compared to a single learner.

It's important to note that one drawback is that there's increased computation time and reduces interpretability. 


## 3.0 Bagging

Bagging is a technique where reuse the same training algorithm several times on different subsets of the training data. 

### 3.1 Algorithm

Given a training dataset D of size N, bagging will generate new training sets D<sub>i</sub> of size M by sampling with replacement from D. Some observations might be repeated in each D<sub>i</sub>. 

If we set M to N, then on average 63.2% of the original dataset D is represented, the rest will be duplicates.

The final step is that we train the classifer C on each C<sub>i</sub> separately. 


## 4.0 Boosting

Boosting is an optimization technique that allows us to combine multiple classifiers to improve classification accuracy. In boosting, none of the classifiers just need to be at least slightly better than chance. 

Boosting involves training classifiers on a subset of the training data that is most informative given the current classifiers. 

### 4.1 Algorithm

The general boosting algorithm first involves fitting a simple model to subsample of the data. Next, we identify misclassified observations (ones that are hard to predict). we focus subsequent learners on these samples to get them right. Lastly, we combine these weak learners to form a more complex but accurate predictor.

### 4.2 Boosting in R

First, we load the required packages: 

``` R
library(mboost)
```

Through the `glmboost()` function, we can fit a generalized linear model. The difference between this function the normal `glm()` function is that the boosted version inherently performs variable selection:

``` R
m.boost <- glmboost(Class ~ Amount + Duration
+ Personal.Female.Single,
family=Binomial(), # needed for classification
data=GermanCredit)
```

Let's plot the convergence of selected coefficients:

``` R
plot(m.boost, ylim=range(coef(m.boost,
which=c("Amount", "Duration"))))
```

The main parameter for tuning is the number of iterations. Using cross-validated estimates of empirical risk, we can find the optimal number. In R, this looks like:

``` R
cv.boost <- cvrisk(m.boost)
```

With `cv.boost`, we can find the optimal number of iterations to prevent overfitting:

``` R
mstop(cv.boost)
```

Let's plot these cross-validated estimates of empirical risk: 

``` R
plot(cv.boost, main="Cross-validated estimates of empirical risk")
```

## 5.0 AdaBoosting

Now, instead of resampling, we can reweight misclassified training examples:

### 5.1 Benefits

Aside from its easy implementation, AdaBoosting is great because it's a simple combination of multiple classifiers. These classifiers can also be different. 

### 5.2 Limits

On the other hand, AdaBoost is sensitive to misclassified points in the training data. 

### 5.3 AdaBoost in R

We begin by loading the required package: 

``` R
library(ada)
```

First, we pick a fixed number of iterations (50 in this case) and feed it along with the training data to fit the AdaBoost model:

``` R
m.ada <- ada(Class ~ .,
data=GermanCredit[inTrain,],
iter=50)
```

And now we evaluate on the test data:

``` R
m.ada.test <- addtest(m.ada,
test.x=GermanCredit[-inTrain,],
test.y=GermanCredit$Class[-inTrain])
```

Now we plot the error on training and testing data:

``` R
plot(m.ada.test, test=TRUE)
```





