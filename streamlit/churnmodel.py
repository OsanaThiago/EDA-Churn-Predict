class ChurnModel(): 
    def __init__(self, pipeline, features, df):
        self.pipeline = pipeline
        self.features = features
        self.df = df

    def predict(self, dados):
        return self.pipeline.predict_proba(dados[self.features])[0, 1]

    def baseline(self):
        return self.df["Exited"].mean()

    def percentil_risco(self, probachurn):
        scores = self.pipeline.predict_proba(self.df[self.features])[:, 1]
        return (scores < probachurn).mean()