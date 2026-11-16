from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_knn(X_train, y_train):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])
    grid = {
        "knn__n_neighbors": [3, 5, 7, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }
    search = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=3), scoring="accuracy")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_