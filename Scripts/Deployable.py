import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(numerical_attributes , categorical_attributes):
    #''''''''' Creating pipelines '''''''''''
    Numerical_pipeline = Pipeline ([
        ("impute", SimpleImputer(strategy="median")),
        ("Standardize" , StandardScaler()),
    ])
    Categorical_pipeline = Pipeline([
        ("Encoder",OneHotEncoder(handle_unknown="ignore")),
    ])

    mypipeline = ColumnTransformer([
        ("numerical" , Numerical_pipeline , numerical_attributes),
        ("categorical" , Categorical_pipeline , categorical_attributes),
    ])
    return mypipeline

if not os.path.exists(MODEL_FILE):
    #model training
    housing = pd.read_csv("housing.csv")
    #Performing split
    housing["income_category"] = pd.cut(housing["median_income"],bins = [0,1.5,3.0,4.5,6.0,np.inf],labels = [1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.2 , random_state = 42)
    for train_index,test_index in split.split(housing , housing["income_category"]):
        train_set = housing.loc[train_index].drop("income_category", axis=1)
        test_set = housing.loc[test_index]
    #Separating data into features and labels
    Features = train_set.drop('median_house_value',axis=1)
    labels = train_set["median_house_value"].copy()
    #Separating data into numerical and categorical attrributes
    numerical_attributes = Features.drop("ocean_proximity", axis=1).columns.tolist()
    categorical_attributes =["ocean_proximity"]
    #build pipeline
    Pipeline = build_pipeline(numerical_attributes , categorical_attributes)
    Final_data = Pipeline.fit_transform(Features)
    #Model
    Model = RandomForestRegressor(random_state=1)
    Model.fit(Final_data , labels)
    #Dumping into pkl file using joblib
    joblib.dump(Model , MODEL_FILE)
    joblib.dump(Pipeline , PIPELINE_FILE)
    #Success
    print("Model has successfully trained")
else:
    Model = joblib.load(MODEL_FILE)
    Pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("test_data.csv")
    transform_input = Pipeline.transform(input_data)
    predictions = Model.predict(transform_input)
    input_data["median_house_value"]=predictions
    input_data.to_csv("output.csv")
    print("Congratulations!Your predications are in output.csv")