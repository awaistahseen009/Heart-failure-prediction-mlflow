from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessing
from build_model import BuildModel
from evaluate import EvaluateModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from datetime import datetime
from sklearn.ensemble import ExtraTreesClassifier
# I have imported all algorithms , so in future if anyone 
# modify this repo can check any algorithm
'''
DATA INGESTION CLASS IS USED TO GET THE DATA AND MAKE IT
READY IN DATAFRAME FORMAT
'''
ingest_data=DataIngestion()
df=ingest_data.get_data('data\heart.csv')
print('***********Data has been ingested***********')
print("---------------------------------------------")
'''
DATA PREPROCESSING CLASS IS USED TO PREPROCESS THE DATA AND MAKE
IT READY FOR THE MODEL
'''
clean_data=DataPreprocessing(df,scale=True)
clean_data.show_shape()
clean_data.remove_null_values()
clean_data.remove_duplicates()
clean_data.remove_outliers()
clean_data.encode_categorical_cols()
clean_data.set_x_y(target="HeartDisease")
clean_data.split_data(test_size=0.1)
x_train, x_test, y_train, y_test =clean_data.get_split_data()
print('***********Data has been cleaned***********')
print("---------------------------------------------")
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name=f"Heart_Disease_Classification_{current_datetime}"
'''
MODEL DEFINITION AND TRAINING
'''
config= {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15,20,25],
    'min_samples_split': [2, 5, 10,15],
}
print('***********Model training starting***********')
print("---------------------------------------------")
rfc = RandomForestClassifier()
set_model=BuildModel(x_train, y_train , model=rfc,hyper_parameter_tuning=True,config=config,return_model=True)
model=set_model.build_model()
set_model.print_scores_and_params()


'''
MODEL EVALUATION
'''
print('***********Model Evaluation Started***********')
print("---------------------------------------------")
evaluate_model=EvaluateModel(model=model, x_test=x_test, y_test=y_test,save_metrics=True,experiment_name=experiment_name ,mlflow_tracking=True)
evaluate_model.print_evaluation_metrics()