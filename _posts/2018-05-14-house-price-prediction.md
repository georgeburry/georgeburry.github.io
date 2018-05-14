---
title: "Pipeline for predicting house prices"
date: 2018-05-14
tags: [pipeline]
header:
  image: "/images/machine-learning-header.jpg"
excerpt: "Predicting Boston house prices using various models"
mathjax: ""
---
[The Jupyter notebook can be accessed here](https://github.com/georgeburry/house-price-prediction/blob/master/house-price-prediction.ipynb)

This is a simple example of how to: take a modest dataset, distill the feature set through exploratory data analysis, and then scale down features and train a variety of regression models via a pipeline. In the end, I select the random forest regression model, which typically yields very good results, and apply a grid search to tune model parameters, before predicting prices.

## Loading in data
In this case we are using the standard Scikit-learn dataset called 'Boston House Prices', which contains features about houses (e.g. location, age, number of rooms, basement area), where the median values of the properties are the target values.

### Notes about the data

Data Set Characteristics:  

    :Number of Instances: 506

    :Number of Attributes: 13 numeric/categorical predictive

    :Median Value (attribute 14) is usually the target

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

I will now load the data into a Pandas dataframe by concatenating the data and the targets from the Boston dataset.

```python
# Load dataset
data = datasets.load_boston()
# Extract feature names, predictors and targets from sklearn dictionary
names = data.feature_names
predictors = data.data
targets = data.target
# Concatenate predictors and targets for easier processing later on. Let's also name the columns.
df = pd.concat([pd.DataFrame(predictors, columns=names), pd.DataFrame(targets, columns=['MEDV'])], axis=1)
```

## Inspecting data
We have everything in one table with all columns labelled, as intended.

```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.00000</td>
      <td>2.31000</td>
      <td>0.00000</td>
      <td>0.53800</td>
      <td>6.57500</td>
      <td>65.20000</td>
      <td>4.09000</td>
      <td>1.00000</td>
      <td>296.00000</td>
      <td>15.30000</td>
      <td>396.90000</td>
      <td>4.98000</td>
      <td>24.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.00000</td>
      <td>7.07000</td>
      <td>0.00000</td>
      <td>0.46900</td>
      <td>6.42100</td>
      <td>78.90000</td>
      <td>4.96710</td>
      <td>2.00000</td>
      <td>242.00000</td>
      <td>17.80000</td>
      <td>396.90000</td>
      <td>9.14000</td>
      <td>21.60000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.00000</td>
      <td>7.07000</td>
      <td>0.00000</td>
      <td>0.46900</td>
      <td>7.18500</td>
      <td>61.10000</td>
      <td>4.96710</td>
      <td>2.00000</td>
      <td>242.00000</td>
      <td>17.80000</td>
      <td>392.83000</td>
      <td>4.03000</td>
      <td>34.70000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.00000</td>
      <td>2.18000</td>
      <td>0.00000</td>
      <td>0.45800</td>
      <td>6.99800</td>
      <td>45.80000</td>
      <td>6.06220</td>
      <td>3.00000</td>
      <td>222.00000</td>
      <td>18.70000</td>
      <td>394.63000</td>
      <td>2.94000</td>
      <td>33.40000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.00000</td>
      <td>2.18000</td>
      <td>0.00000</td>
      <td>0.45800</td>
      <td>7.14700</td>
      <td>54.20000</td>
      <td>6.06220</td>
      <td>3.00000</td>
      <td>222.00000</td>
      <td>18.70000</td>
      <td>396.90000</td>
      <td>5.33000</td>
      <td>36.20000</td>
    </tr>
  </tbody>
</table>
</div>

Lets confirm what type of values we have (should all be floating values).

```python
df.dtypes.value_counts()
```

    float64    14
    dtype: int64

Lets summarize each attribute. From a glance we could expect some of the data to be quite skewed.

```python
df.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
      <td>506.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.59376</td>
      <td>11.36364</td>
      <td>11.13678</td>
      <td>0.06917</td>
      <td>0.55470</td>
      <td>6.28463</td>
      <td>68.57490</td>
      <td>3.79504</td>
      <td>9.54941</td>
      <td>408.23715</td>
      <td>18.45553</td>
      <td>356.67403</td>
      <td>12.65306</td>
      <td>22.53281</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.59678</td>
      <td>23.32245</td>
      <td>6.86035</td>
      <td>0.25399</td>
      <td>0.11588</td>
      <td>0.70262</td>
      <td>28.14886</td>
      <td>2.10571</td>
      <td>8.70726</td>
      <td>168.53712</td>
      <td>2.16495</td>
      <td>91.29486</td>
      <td>7.14106</td>
      <td>9.19710</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00632</td>
      <td>0.00000</td>
      <td>0.46000</td>
      <td>0.00000</td>
      <td>0.38500</td>
      <td>3.56100</td>
      <td>2.90000</td>
      <td>1.12960</td>
      <td>1.00000</td>
      <td>187.00000</td>
      <td>12.60000</td>
      <td>0.32000</td>
      <td>1.73000</td>
      <td>5.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.08204</td>
      <td>0.00000</td>
      <td>5.19000</td>
      <td>0.00000</td>
      <td>0.44900</td>
      <td>5.88550</td>
      <td>45.02500</td>
      <td>2.10018</td>
      <td>4.00000</td>
      <td>279.00000</td>
      <td>17.40000</td>
      <td>375.37750</td>
      <td>6.95000</td>
      <td>17.02500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.25651</td>
      <td>0.00000</td>
      <td>9.69000</td>
      <td>0.00000</td>
      <td>0.53800</td>
      <td>6.20850</td>
      <td>77.50000</td>
      <td>3.20745</td>
      <td>5.00000</td>
      <td>330.00000</td>
      <td>19.05000</td>
      <td>391.44000</td>
      <td>11.36000</td>
      <td>21.20000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.64742</td>
      <td>12.50000</td>
      <td>18.10000</td>
      <td>0.00000</td>
      <td>0.62400</td>
      <td>6.62350</td>
      <td>94.07500</td>
      <td>5.18843</td>
      <td>24.00000</td>
      <td>666.00000</td>
      <td>20.20000</td>
      <td>396.22500</td>
      <td>16.95500</td>
      <td>25.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.97620</td>
      <td>100.00000</td>
      <td>27.74000</td>
      <td>1.00000</td>
      <td>0.87100</td>
      <td>8.78000</td>
      <td>100.00000</td>
      <td>12.12650</td>
      <td>24.00000</td>
      <td>711.00000</td>
      <td>22.00000</td>
      <td>396.90000</td>
      <td>37.97000</td>
      <td>50.00000</td>
    </tr>
  </tbody>
</table>
</div>

We can also plot histograms for each column to get a clear idea of the shape of the data. It might be a good idea to redistribute some of the data. I will skip over it in this case.

```python
df.hist(figsize=(10, 10), bins=20)
plt.show()
```

![png](/images/house-price-prediction_files/house-price-prediction_12_0.png)

We need to make sure we don't have any Nan values in our dataset, otherwise the ML algorithms will fail.

```python
df.isna().sum()
```

    CRIM       0
    ZN         0
    INDUS      0
    CHAS       0
    NOX        0
    RM         0
    AGE        0
    DIS        0
    RAD        0
    TAX        0
    PTRATIO    0
    B          0
    LSTAT      0
    MEDV       0
    dtype: int64

Good, now let's see if we can find strong correlations between columns.

```python
df_corr = df.corr().abs()
filter = (df_corr == 1) | (df_corr < 0.75)
df_corr[filter] = 0
df_corr

f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df_corr, annot=True, annot_kws={'size': 14},
            cmap=sns.diverging_palette(220, 20, n=11))
```

![png](/images/house-price-prediction_files/house-price-prediction_16_1.png)

## Dimensionality reduction
From the previous plot, we were able to ascertain that the 'NOX' and the 'RAD' columns are strongly correlated with other columns. Therefore, let's go ahead and drop them from our dataframe.

```python
cols_corr_manual = ['NOX', 'RAD']
df = df.drop(columns=cols_corr_manual)
```

## Setting up train and test datasets
Now we are ready to peel of the last column and assign it to the target (y) variable and the rest of the data can be assigned to the predictor (X) variable. After that, we use the sklearn train_test_split function to create train and test datasets for both X and y.

```python
# Train-set predictors/targets
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
```

## Initialising models, setting up a pipeline, training and predicting
For this part, I would like a list of regression models, then I will create a pipeline that scales the data, reduces feature dimensionality (via PCA) and trains the model, all in one go. We can loop through each model, inserting it into the pipeline, and then finally we can see how the results for each of the models compare.

```python
# Lets append tuples to the list that contain both the name of the model and the model itself
# This is what the pipeline expects
models = []
models.append( ('LR', LinearRegression()) )
models.append( ('Lasso', Lasso()) )
models.append( ('ElasticNet', ElasticNet()) )
models.append( ('KNN', KNeighborsRegressor()) )
models.append( ('CART', DecisionTreeRegressor()) )
models.append( ('SVR', SVR()) )
```

```python
# Now we can loop through the models and run the pipeline each time
for name, model in models:
    pipelined_model = Pipeline([('minmax', MinMaxScaler()),
                     ('pca', PCA(n_components = 3)),
                     (name, model)])

    # Train model
    pipelined_model.fit(X_train, y_train)

    # Make predictions on the test-set
    y_hat = pipelined_model.predict(X_test)

    # Calculate error
    RMSE = np.sqrt(mean_squared_error(y_test, y_hat))
    print('Model: ', name, ' | RMSE: ', RMSE)
    print('----------------')

    # I also like to save the models each time as a matter of habit
    joblib.dump(pipelined_model, '{}_model.pkl'.format(name))
```

    Model:  LR  | RMSE:  7.15970109322
    ----------------
    Model:  Lasso  | RMSE:  7.57962692126
    ----------------
    Model:  ElasticNet  | RMSE:  8.1579157098
    ----------------
    Model:  KNN  | RMSE:  6.36588579546
    ----------------
    Model:  CART  | RMSE:  9.45087919969
    ----------------
    Model:  SVR  | RMSE:  7.60464954267
    ----------------

We will now run the pipeline just once with yet another model - the RandomForest regressor - and incorporate a gridsearch to tune the model parameters.

```python
# The pipeline as before, but this time the model type is static
pipelined_model = Pipeline([('minmax', MinMaxScaler()),
                 ('pca', PCA(n_components = 3)),
                 ('RF', RandomForestRegressor())])

grid = {
        'RF__n_estimators': [100, 200, 300, 400, 500],
        'RF__max_depth': [10, 20, 30, 40, 50],
        'RF__min_samples_leaf': [10, 15, 20, 25, 30]
        }

# Run cross-validation & discover best combination of params, with 10 folds of cross-validation
grid_search_cv = GridSearchCV(pipelined_model,
                              grid,
                              cv=KFold(n_splits=10, random_state=7),
                              scoring='neg_mean_squared_error',
                              n_jobs = -1)

results = grid_search_cv.fit(X_train, y_train)

print(results.best_score_)
print(results.best_params_)
```

Now we can create a new instance of the pipeline and pass the 'best_params_' of the grid search directly into the regressor.

```python
pipelined_model = Pipeline([('minmax', MinMaxScaler()),
                 ('pca', PCA(n_components = 3)),
                 ('RF', RandomForestRegressor(n_estimators=results.best_params_['RF__n_estimators'],
                                              max_depth=results.best_params_['RF__max_depth'],
                                              min_samples_leaf=results.best_params_['RF__min_samples_leaf']))])

pipelined_model.fit(X_train, y_train)
```

Once more, we make predictions using the test set.

```python
# Lets make predictions on the test-set
y_hat = pipelined_model.predict(X_test)

# Calc error
RMSE = np.sqrt(mean_squared_error(y_test, y_hat))
print('Model: Random forest', ' | ', 'RMSE: ', RMSE)
```

    Model: Random forest  |  RMSE:  6.31066783749

Turns out that our RMSE score wins out by a very tight margin. Let's bare in mind that we off by about $6000 US here, because the target values are counted in the 1000s.
