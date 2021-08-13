---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Data Preparation & Classification

Before getting to our prediction algorithm, our data must go through different processes, in different subsets.
The order of such processes is often misunderstood.
For this reason, we are going to start from a picture summing up the whole process, and go through it step by step.

![data_pipeline](images/flowchart.png)

1. **Data Cleaning**: Here is where we handle *missing* values from our dataset. In our case we also perform an *Ordinal Encoding*, to ease later steps.
2. **Data Split**: At this point, we take away a small portion of data and put it aside. This small portion will simulate newly gathered data, and it will be essential to test the generalization power of our final model.
3. **Training Path**:
    1. **Oversampling**: We use SMOTE to oversample **only training data**. Since the real world is unbalanced, if we were to balance also the test set, we would cheat, and offer a distorted vision of what our model would expect.
    2. **Scaling**: Some ML do not have any problem with our data having different scales (i.e. Trees), but some of them do (i.e. SVM). In order to have a unique pipeline that works with every kind of algorithm, we will always apply scaling.
    2. **One-Hot-Encoding**: Ordinal encoding can put our data on different levels that are not really there (i.e. when mapping colors with numbers, there is no reason of assigning 1 to 'black' or 'white'). For this reason, we apply one-hot-encoding. This will avoid the aforementioned problem, but has the drawback of increasing the dimensionality of our data (creating a sparse encoding matrix).

The horizontal arrows mean that we apply the same process, but we only transform data using statistics computed on the training set. i.e. when performing a normalization (scaling), we scale test data using mean and variance computed on training data.

The missing part of the chart deals with the actual classification task, which we will see in a while.
Now, let's prepare our data.

+++

## Training-Test Dataset Split 

To make the code more compact and readable, we are going to use sklearn's pipeline object to create a reusable pipeline of actions.

The first step is to put aside a small portion of the dataset, and call it our *test data*.

```{code-cell} ipython3
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./dataset/online_shoppers_intention.csv')
df_train, df_test = train_test_split(df, test_size=0.2)

x_train, y_train = df_train.drop(columns='Revenue'), df_train['Revenue']
print(f'training data shape: {df_train.shape}\t\ttest data shape: {df_test.shape}')
df.head()
```

## Column Transformer 
For all those actions that require statistics computed column-wise, we use the *ColumnTransformer* object, in which we can insert all those procedures like *Encoding* and *Scaling*.

```{code-cell} ipython3
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder

textual_columns = ['Month', 'VisitorType', 'Weekend']
categorical_columns = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
numerical_columns = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']

column_transformer = ColumnTransformer([
        ('OrdinalEncoder', OrdinalEncoder(), textual_columns),
#         ('MinMaxScaler', MinMaxScaler(), numerical_columns),
#         ('OneHotEncoder', OneHotEncoder(), categorical_columns),
    ],
    remainder='passthrough'
)
```

## Pipeline
We can then inset the column transformer inside a pipeline alongisde the *oversampling* technique that we desire, and the classification algorithm (here we use a *Random Forest* as an example)

```{code-cell} ipython3
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier

categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
categorical_indices = [c in categorical_features for c in df_train.columns]

clf = Pipeline(
    steps=[
        ('ColumnTransformer', column_transformer),
        ('SMOTENC', SMOTENC(categorical_features=categorical_indices)),
#         ('OneHotEncoder', one_hot_encoder),
        ('Classifier', RandomForestClassifier())
    ])
```

## GridSearch & CrossValidation

*GridSearch* is one of many approaches to *hyperparameter optimization*. It is an exaustive search of a predefined subset of hyperparameters (values for continuos parameters are implicitly discretized). The algorithm is then trained with each n-uple in the cartesian product of the sets of each parameter, and is evaluated on a held-out validation set. 

Since we are also doing *CrossValidation*, each hyperparameter configuration is evaluated on each of the k folds in which we split our training set.

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV


param_grid = [
    {
        'Classifier__random_state': [42],
    }
]

# linear_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=6).fit(x_train, y_train)
linear_search.cv_results_
```

```{code-cell} ipython3

```
