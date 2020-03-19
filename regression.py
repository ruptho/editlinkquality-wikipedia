import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler


class SMWrapper(BaseEstimator, RegressorMixin):
    """
    A universal sklearn-style wrapper for statsmodels regressors
    taken from: https://stackoverflow.com/a/48949667
    """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X, False)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X).round()


def shuffle_and_scale(data_x, data_y, shuffle, scaler):
    X, y = np.asarray(data_x), np.asarray(data_y)
    if shuffle:
        idx = np.random.permutation(len(data_y))
        X, y = X[idx], y[idx]

    if scaler is not None:
        X = scaler.fit(X).transform(X)
    return X, y


def perform_logistic_regression_control_sm(data_x, data_y, shuffle=True, feature_names=None, regularize=None,
                                           scaler=None, prepend=False):
    # Uses statsmodels
    X, y = shuffle_and_scale(data_x, data_y, shuffle, scaler)
    est = sm.Logit(y, sm.add_constant(X, prepend))  # add intercept
    if feature_names:
        est.exog_names[:] = ['Intercept'] + feature_names if prepend else feature_names + ['Intercept']
    result = est.fit(full_output=True) if regularize is None else est.fit_regularized(method=regularize,
                                                                                      full_output=True)
    return result


def perform_logistic_regression_prediction(data_x, data_y, use_sm=True, scoring='f1', scaler=RobustScaler(),
                                           shuffle=False, max_iter=100, penalty='l1'):
    # Uses statsmodels (use_sm = True) or scikit learn (use_sm = False) logistic regression
    X, y = shuffle_and_scale(data_x, data_y, shuffle, scaler=scaler)
    solver = 'lbfgs' if penalty == 'l2' else 'liblinear'
    return cross_val_score(
        SMWrapper(sm.Logit) if use_sm else LogisticRegression(solver=solver, penalty=penalty, max_iter=max_iter,
                                                              multi_class='auto'), X, y, cv=10, scoring=scoring)
