import joblib

svmmodel = joblib.load('imports/nbmodel.joblib')
nbmodel = joblib.load('imports/svmmodel.joblib')
tf = joblib.load('imports/customer_review_vector.joblib')