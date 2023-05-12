import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras import optimizers, losses, metrics

class NNModelBalle:
    """
    Neural network model for ball tracking
    """

    def __init__(self, **kwargs) -> None:
        """
        Constructor of the class

        Parameters
        ----------
        **kwargs
            model_path: str
                path .h5 to the model
            layers: list
                list of layers
        """
        try:
            self.model_path = kwargs['model_path']
            self.model = tf.keras.models.load_model(self.model_path)
            self.summary_model = self.model.summary()
            self.layers = self.model.layers
            return
        except KeyError:
            self.model_path = None

        try:
            self.layers = kwargs['layers']
            self.model = tf.keras.models.Sequential(self.layers)
            self.summary_model = self.model.summary()
            self.model_path = None
        except KeyError:
            raise KeyError("No layers given")

    @classmethod
    def load_model_from_path(cls, model_path: str) -> 'NNModelBalle':
        """
        Load a model from a path

        Parameters
        ---------- 
        model_path: str 
            path .h5 to the model

        Returns
        -------
        ModelJoueur
            the model loaded
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError("No model found at {}".format(model_path))
        return cls(model_path=model_path)

    @classmethod
    def load_model_from_layers(cls, layers: list) -> 'NNModelBalle':
        """
        Load a model from a list of layers

        Parameters
        ---------- 
        layers: list 
            list of layers

        Returns
        -------
        ModelJoueur
            the model loaded
        """
        if len(layers) == 0:
            raise ValueError("No layers given")
        return cls(layers=layers)

    def save_model(self, model_path: str) -> None:
        """
        Save the model to a path

        Parameters
        ---------- 
        model_path: str 
            path .h5 to the model

        Returns
        -------
        None
        """
        self.model.save(model_path)
        return
    
    def normalize(self, X):
        """
        Normalize the input

        Parameters
        ----------
        X: np.array
            input

        Returns
        -------
        new_X: np.array
            normalized input
        """
        scaler = StandardScaler()
        new_X = scaler.fit_transform(X)
        return new_X
    
    def label_encoder(self, y, possible_outputs: list = [ 2,  3,  4,  6,  7,  8, 10, 11, 12]):
        """
        Encode the labels

        Parameters
        ---------- 
        y: np.array
            labels
        possible_outputs: list
            possible outputs

        Returns
        -------
        new_y: np.array
            encoded labels
        """
        encoder = LabelEncoder()
        encoder.fit(possible_outputs)
        new_y = encoder.transform(y)
        return new_y

    def split_train_test(self, X, y, random_state: int = 42, test_size: float = 0.2):
        """
        Split the data into train and test

        Parameters
        ---------- 
        X: np.array
            input
        y: np.array
            labels
        random_state: int
            random state
        test_size: float
            size of the test
        shape_frame: tuple
            shape of the frame
        nb_frame: int
            number of frame

        Returns
        -------
        X_train: np.array
            train set
        X_test: np.array
            test set
        y_train: np.array
            train labels
        y_test: np.array
            test labels
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, random_state: int = 1234, learning_rate: float = 0.001,
                  epochs: int = 50, batch_size: int = 128, verbose: int = 1):
        """
        Train the model

        Parameters
        ----------
        X_train: np.array
            input train
        y_train: np.array
            output train
        random_state: int
            random state
        learning_rate: float
            learning rate
        epochs: int
            number of epochs
        batch_size: int
            size of the batch
        verbose: int
            verbose

        Returns
        -------
        history: History
            history of the training
        """
        tf.random.set_seed(random_state)
        # checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(
        #         checkpoint_name, monitor='val_loss', verbose=verbose, save_best_only=True, mode='auto')
        # callbacks_list = [checkpoint]
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,  # données transmises pour une session
            # validation_split=0.2,
            # callbacks=callbacks_list
        )
        return history

    def evaluate(self, X_input, y_input):
        """
        Evaluate the model

        Parameters
        ----------
        X_in: np.array
            coordinates of the ball
        y_in: np.array
            input labels

        Returns
        -------
        tup: tuple
            tuple of loss and accuracy
        """
        tup = self.model.evaluate(X_input, y_input)
        print("Loss: ", tup[0])
        print("Accuracy: ", tup[1])
        return tup

    def predict(self, X_input):
        """
        Predict the class of the images

        Parameters
        ----------
        X_input: np.array
            input of coordinates of the ball

        Returns
        -------
        y_pred: list
            list of predicted classes
        """
        scaler = StandardScaler()
        X_normal = scaler.fit_transform(X_input)
        pred = self.model.predict(X_normal)
        y_pred = np.argmax(pred, axis=1)
        return y_pred