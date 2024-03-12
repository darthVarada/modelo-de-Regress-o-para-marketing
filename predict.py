import pickle
import pandas as pd 


class SalesPredictor:
    def __init__(self, model_file):
        self.model_file = model_file
        self.model = self.load_model()

    def load_model(self):
        with open(self.model_file, "rb") as file:
            model = pickle.load(file)
        return model

    def make_predictions(self, data):
        predictions = self.model.predict(data)
        return predictions

    def write_results(self, predictions):
        print("Predictions:")
        print(predictions)

    def run(self):
        input_data = pd.DataFrame({
            'facebook': [100, 200, 300],
            'youtube': [50, 150, 250],
            'newspaper': [20, 30, 40]
        })
        predictions = self.make_predictions(data=input_data)
        self.write_results(predictions=predictions)


if __name__ == "__main__":
    predictor = SalesPredictor(model_file="trained_classifier.pkl")
    predictor.run()
