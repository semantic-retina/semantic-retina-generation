from sklearn.metrics import cohen_kappa_score


def quadratic_kappa(actual, predicted):
    return cohen_kappa_score(predicted, actual, weights="quadratic")
