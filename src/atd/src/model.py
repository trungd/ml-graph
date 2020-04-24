class Model:
    def __init__(self, params, dataset):
        pass
    
    def fit(self, X, y, sample_weight=None):
        print(y)

    def score(self, X, y, metric="acc"):
        if metric == "acc":
            return None