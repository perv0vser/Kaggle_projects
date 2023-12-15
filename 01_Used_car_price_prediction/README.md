# Used car price prediction

## Files

- *Used_car_price_prediction.ipynb* - project notebook
- *train_analysis.html* - interactive train data profiling report
- *test_analysis.html* - interactive test data profiling report
- *train_test_compare_analysis.html* - interactive train/test data comparison report

## Data

- *train.csv* - training dataset (~440000)
- *test.csv* - testing dataset (~110000)

**Features**
- *date:* year of manufacturing
- *make:* manufacturer's name
- *model:* car model
- *trim:* model's options
- *body:* body type
- *transmission:* transmission type
- *vin:* vehicle identification number
- *state:* country state
- *condition:* car condition grade
- *odometer:* car mileage
- *color:* body color
- *interior:* interior color
- *seller:* car seller
- *saledate:* date of selling

**Target**
- *sellingprice:* price of selling

## Description

Developing a model to predict the selling price of a used car. This is a regression problem. Using MAPE (mean percentage absolute error) as a metric, since it is easy to interpret. There is a large dispersion in different classes cars' prices. So in units of measurement the value of the metric would be less clear.

## Stack

`Python`, `Numpy`, `Pandas`, `Seaborn`, `Matplotlib`, `Scikit-learn`, `Sweetviz`, `Fuzzywuzzy`, `Re`, `Phik`, `Vininfo`, `Shap`, `Optuna`, `CatBoost`, `Streamlit`

## Main steps

- **Data preprocessing and analysis**

Missing numeric and categorical values were filled in with medians and modes grouped by several other features (like manufacturer, model and year of manufacturing). A lot of hidden duplicates were detected and corrected due to fuzzy search instruments based on Levenshtein distance calculation (fuzzywuzzy library) and regular expressions which helped to reduce cardinality of some categorical features. Using data from VIN numders helped to find missing values in the main categorical features like "make" and "model" (vininfo library) which helped a lot im more precise fitting.
Some quantity of anomalies and outliers was deleted from training dataset (like unreasonably high price or price close to zero, cars manufactured later then selled, very old cars which quantity was too small to fit a model properly etc.) and this affected models' metrics very positively. Several categorical features with high cardinality could be a problem for one-hot encoding and a risk of overfitting so for linear models and RandomForest fitting their cardinality was decreased to a reasonable level, which made pipelines with this models go much faster and more productive.
All the numeric values distributions are not normal. "Sellingprice" (target) and "odometer" values distributions have long right tails, "year" values distribution has a long left tail and together with "condition" has multiple peaks as they both are not really continuous values. Using logarithm could help with it.
Phik correlation matrix shows not very high values of linear connection between features and target which makes us think of a nonlinear nature of the relationships. The main features connected to the selling price are "model" (0.49), "make" (0.46), "odometer" (0.43), "condition" (0.40) and "year" (0.39) which is logical.

- **Baseline creating**

Lasso linear model which L1 regularization should have helped to determine less informative features scored only 61% MAPE value which is reasonable due to nonlinear nature of the relationships mentioned above and confirmed with RandomForest model much better score of 19% MAPE (interesting that without numeric features scaling RandomForest scored only 68%). And the best result - 17% MAPE from CatBoost. First 2 models were used as more simple and to help with feature selection, last one as it can fit the features with minimum preprocessing required and provide a decent score out of the box.

- **Feature engineering and selection**

Added "age" feature which is equal to year, but doesn't depend on the year of sale (there are sales dated with 2014 and 2015 years represented in the dataset); "cluster" (ranged features to 5 clusters with Kmeans and used as a new feature) which appeared to be not a very informative feature; "odoyear" (annual mileage) which also didn't improve any model; "country" and "region" of manufacturing extracted from VIN and supposed to give additional clue about price level when "make"/"model" are new to a model (didn't manage to register any improvement, maybe because most of "make"/"model" from test dataset were learned from the training one). Also made manual target encoding for "make" and "model" features which in combintion with One-Hot or Ordinal encoding of the other categorical features provided better scores from Linear model and Random forest than CatBoostEncoder used with all the categorical features. Tried Mutual info regression, Variance Inflation Factor and PCA based methods for the feature selection and Mutual info regression provided the best result of them. All the feature importance analysis methods were confirming that "transmission" is the less informative feature (which is logical as one value has 97% part), "color" and "interior" as well (which is logical too as body and interior colors doesn't seem to be main car price components).

- **Modeling with hyperparameters optimization**

Tried ElasticNet model with "alpha" parameter search via GridSearchCV (53% MAPE), didn't try polynomial features in order to describe nonlinear relations better (maybe next time), RandomForest with multiple parameters adjustment via RandomizedSearchCV to make it not so long (17% MAPE) and CatBoost with multiple parameters optimization via OptunaSearchCV on modified features with manual target encoding for "make" and "model" (15.7% MAPE) and on original features (14.9% MAPE), explained feature importance via SHAP, then dropped 3 less informative features (described above) and scored 14.67% MAPE. And due to the target distribution is not normal I used logarithm in TransformedTargetRegressor to help model and improved the best model's result to 13.43% MAPE. Bagging technique with 2 or 3 gradient boosting models could be used for a further metric optimization (for competition purposes), but it is not very practical for real cases.

- **Testing the best model**

The best model scored 13.59% MAPE on the testing dataset which is really close to it's performance on the training data which means there is no overfitting. It confirms that CatBoost does enough data optimization automatically and the best result was achieved due to a correct raw data preprocessing (missing values filling, duplicates correction, anomalies and outliers elimination) and hyperparameters optimization via Optuna. 
