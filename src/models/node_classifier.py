import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .node2vec import Node2Vec


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


class NodeClassifier(BaseEstimator):
    def __init__(self, params, dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset
        self.emb = Node2Vec(params, dataset)
        self.mlb = MultiLabelBinarizer(range(dataset.num_classes))
        self.clf = TopKRanker(LogisticRegression(solver="lbfgs"))

    def score(self, X, y, metric):
        y_pred = self.predict(X, y)
        assert len(y_pred) == len(y)
        assert all(len(p) == len(r) for p, r in zip(y_pred, y))
        if metric[:3] == "f1_":
            return f1_score(self.mlb.fit_transform(y), self.mlb.fit_transform(y_pred), average=metric[3:])
        elif metric == "acc":
            return accuracy_score(self.mlb.fit_transform(y), self.mlb.fit_transform(y_pred))

    def fit(self, X, y=None, **fit_params):
        X = self.emb.fit_transform(X)
        y = self.mlb.fit_transform(y)
        self.clf.fit(X, y)

    def predict(self, X, y=None):
        X = self.emb.transform(X)
        return self.clf.predict(X, [len(ls) for ls in y])