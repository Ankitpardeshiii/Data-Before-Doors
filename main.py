import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#''''''''' Reading data ''''''''''
housing = pd.read_csv("housing.csv")
#''''''''' Spliting data into train and test sets '''''''''
housing["income_category"]=pd.cut(housing["median_income"],bins = [0,1.5,3.0,4.5,6.0,np.inf],labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=1)
for train_index,test_index in split.split(housing , housing["income_category"]):
    train_set = housing.iloc[train_index].drop(["income_category"],axis = 1)
    test_set = housing.iloc[test_index].drop(["income_category"],axis = 1)
#''''''''' Working on copy of the train data ''''''''''
data = train_set.copy()
#''''''''' Separating features and labels ''''''''''
Features = data.drop("median_house_value",axis = 1)
Labels = data["median_house_value"].copy()
#''''''''' Separating numerical and categorical attributes ''''''''''
numerical_attributes = Features.drop("ocean_proximity",axis=1).columns.tolist()
categorical_attributes = ["ocean_proximity"]
#''''''''' Creating pipelines '''''''''''
Numerical_pipeline = Pipeline ([
    ("impute", SimpleImputer(strategy="median")),
    ("Standardize" , StandardScaler()),
])
Categorical_pipeline = Pipeline([
    ("Encoder",OneHotEncoder()),
])

mypipeline = ColumnTransformer([
    ("numerical" , Numerical_pipeline , numerical_attributes),
    ("categorical" , Categorical_pipeline , categorical_attributes),
])
#''''''''''' Transforming data ''''''''''''
Final_data = mypipeline.fit_transform(Features)
#''''''''''' Training models '''''''''''
#Decision Tree
decision_tree = DecisionTreeRegressor()
decision_tree.fit(Final_data , Labels)
decision_tree_predictions = decision_tree.predict(Final_data)
decision_tree_error = -cross_val_score(decision_tree , Final_data , Labels , scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(decision_tree_error).describe())
#Random forest
Random_forest = RandomForestRegressor()
Random_forest.fit(Final_data , Labels)
Random_forest_predictions = Random_forest.predict(Final_data)
Random_forest_error = -cross_val_score(Random_forest , Final_data , Labels , scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(Random_forest_error).describe())
#Linear Regression
Linear_Regression = LinearRegression()
Linear_Regression.fit(Final_data , Labels)
Linear_Regression_predictions = Linear_Regression.predict(Final_data)
Linear_Regression_error = -cross_val_score(Linear_Regression, Final_data , Labels , scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(Linear_Regression_error).describe())