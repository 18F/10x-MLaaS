class PredictionManager():

    def __init__(self, data_source, ml_algorithm):
        self.data_mgr = data_source
        self.ml_mgr = ml_algorithm
        self.data = None
        self.prediction = None

    def get_raw_data(self):
        self.data = self.data_mgr.get_data()

    def predict(self):
        # This will assume raw_data is available, if not it will get raw_data in dataframe
        if self.data is None:
            print("Getting new data")
            self.data = self.data_mgr.get_data()
        # We can then apply algorithm to prediction
        if self.data is not None:
            # return dataframe with prediction
            print("Ready to do prediction")
            self.prediction = self.ml_mgr.predict(self.data)

        return self.prediction
