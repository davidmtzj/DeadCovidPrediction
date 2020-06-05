import joblib
import numpy as np

class MLDeadPredictor:
    def __init__(self, X_values: []):
        self.X = np.array(X_values)
        self.X = self.X.reshape(1,-1)


    def predictSGD(self):

        model = joblib.load('models/covid_SGDClassifier_model.pkl')
        prediction = model.predict(self.X)
        print("Prediction:", prediction)
        return prediction