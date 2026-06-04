import joblib

model = joblib.load("model.pkl")

print("Features:", model.n_features_in_)
print("Classes:", model.classes_)