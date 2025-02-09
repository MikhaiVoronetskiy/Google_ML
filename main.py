import joblib


def predict(X):
    model = joblib.load('model.pkl')
    return model.predict(X)



