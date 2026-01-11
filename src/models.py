from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier


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



# xgboost handles is mathematically optimized and handles clinical data imbalances 

def train_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

# model evaluation

def evaluation_of_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return classification_report(y_test, predictions)

# confusion matrix

def confusion_matrix_display(model, X_test, y_test, ax, cmap, title):
    display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title, fontsize=12)
    return display