from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Model 1: Logistic regression. -> this is famous one for small medical data, I chose this linear classifier
#  model because it calculates death_event as a linear combination of its features  
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

#random forest is the good choice when we need a strong accuracy and it is ideal for small medical data, like ours

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluation_of_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return classification_report(y_test, predictions)
