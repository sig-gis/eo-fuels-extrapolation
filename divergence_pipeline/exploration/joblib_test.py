import joblib
import sklearn

file_name = r'C:\Users\edalt\RD_Fuels\eo-fuels-extrapolation\data\rf_pyrome_26.joblib'  # Replace with your file path
loaded_object = joblib.load(file_name)

print(loaded_object)
print(type(loaded_object))
print(loaded_object.get_params())
print(loaded_object.classes_)
print(loaded_object.n_features_in_)
print(loaded_object.feature_names_in_)