import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def fit_and_predict(train_scores, train_labels, test_scores):
    X = np.asarray(train_scores, dtype=float)
    y = np.asarray(train_labels, dtype=int)
    Xt = np.asarray(test_scores, dtype=float)

    # QDA with regularization for stability
    clf = QuadraticDiscriminantAnalysis(reg_param=0.2)
    clf.fit(X, y)
    pred = clf.predict(Xt)
    return pred.astype(int).tolist()