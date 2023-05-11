from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
import joblib

class RFModelBalle:
    """
    Random Forest Classifier for the ball tracking
    """
    def __init__(self, **kwargs) -> None:
        """
        Parameters
        ----------
        n_estimators : int
            Number of estimators for the random forest classifier
        
        **kwargs : dict
            Keyword arguments
        Keyword Arguments
        """
        try:
            self.path = kwargs["path"]
            self.model = joblib.load(self.path)
            return
        except KeyError:
            pass

        try:
            max_depth = kwargs["max_depth"]
            n_estimators = kwargs["n_estimators"]
            self.model = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators)
            self.path = None
        except KeyError:
            pass

    @classmethod
    def load_model_from_path(cls, path: str):
        """
        Parameters
        ----------
        path : str
            Path to the model
        
        Returns
        -------
        ModelBalle
            The model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        return cls(path = path)
    
    @classmethod
    def construct_model(cls, max_depth: int = 8, n_estimators: int = 20):
        """
        Parameters
        ----------
        n_estimators : int, optional
            Number of estimators for the random forest classifier, by default 100

        Returns
        -------
        ModelBalle
            The model
        """
        return cls(max_depth = max_depth, n_estimators = n_estimators)
    
    def save_model(self, path: str):
        """
        Parameters
        ----------
        path : str
            Path to save the model .joblib
        """
        joblib.dump(self.model, path)
        return
    
    def train_test_split(self, X, y, test_size: int = 0.2, random_state: int = 42):
        """
        Parameters
        ----------
        X : np.ndarray
            The data
        y : np.ndarray
            The labels
        test_size : int, optional
            The size of the test set, by default 0.2
        random_state : int, optional
            The random state, by default 42

        Returns
        -------
        X_train, X_test, y_train, y_test
            The train and test sets
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train, y_train):
        """
        Parameters
        ----------
        X_train : np.ndarray
            The data
        y_train : np.ndarray
            The labels

        Returns
        -------
        None
        """
        self.model.fit(X_train, y_train)
        return
    
    def evaluate(self, X_test, y_test):
        """
        Parameters
        ----------
        X_test : np.ndarray
        y_test : np.ndarray

        Returns
        -------
        float
        """
        return self.model.score(X_test, y_test)
    
    def predict(self, X):
        """
        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """
        return self.model.predict(X)
        