import tensorflow as tf
import os
from ..video.Image import Image
from ..video.Video import Video
import cv2
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np


class ModelJoueurConvolution:
    """
    Convolutional neural network model for the player
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
    def load_model_from_path(cls, model_path: str) -> 'ModelJoueurConvolution':
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
    def load_model_from_layers(cls, layers: list) -> 'ModelJoueurConvolution':
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

    def analyse_sequence_frame(self, path_data_frame: str, nb_frame: int = 15):
        """
        Analyse a sequence of frames

        Parameters
        ---------- 
        path_data_frame: str 
            path to the data frame
        nb_frame: int
            number of frame

        Returns
        -------
        list_videos: list
            list of videos
        y: list
            list of labels
        """
        if not os.path.exists(path_data_frame):
            raise FileNotFoundError("No data frame found at {}".format(path_data_frame))
        list_videos = []
        y = []
        for dirpath, dirnames, _ in os.walk(path_data_frame):
            for dir_type in dirnames:
                path_fol = os.path.join(dirpath, dir_type)
                for file in os.listdir(path_fol):
                    path_fol_img = os.path.join(path_fol, file)
                    if os.path.isdir(path_fol_img) and file == 'images':
                        frames = []
                        for file_img in os.listdir(path_fol_img):
                            path_img = os.path.join(path_fol_img, file_img)
                            img_obj = Image.load_image(cv2.IMREAD_COLOR, path_img)
                            img = img_obj.img
                            frames.append(img)
                        if (len(frames) == nb_frame):
                            output_res = path_img.split('\\')[1]
                            vid = Video.read_video_from_frames(frames)
                            list_videos.append(vid)
                            y.append(output_res)
        return list_videos, y
    
    def label_encoder(self, y):
        """
        Encode the labels

        Parameters
        ---------- 
        y: np.array
            labels

        Returns
        -------
        new_y: np.array
            encoded labels
        """
        encoder = LabelEncoder()
        new_y = encoder.fit_transform(y)
        return new_y

    def split_train_test(self, list_videos, y, random_state: int = 42, test_size: float = 0.2,
                             shape_frame: tuple = (50, 50, 3), nb_frame: int = 15):
        """
        Split the data into train and test

        Parameters
        ---------- 
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
        train_size = int(len(list_videos) * (1 - test_size))
        test_size = len(list_videos) - train_size
        # Split video train, test
        random.seed(random_state)
        vids_train = random.sample(list_videos, k=train_size)
        vids_test = [vid for vid in list_videos if vid not in vids_train]
        y_train = [y[i] for i in range(len(y)) if list_videos[i] in vids_train]
        y_test = [y[i] for i in range(len(y)) if list_videos[i] in vids_test]

        y_train = np.array(y_train, dtype=int)
        y_test = np.array(y_test, dtype=int)

        # Split input X en train, test set
        X_train = np.zeros((train_size, shape_frame[0], shape_frame[1]*nb_frame, shape_frame[2]), dtype=int)
        for i, vid in enumerate(vids_train):
            for j, frame in enumerate(vid.frames):
                X_train[i, :, shape_frame[1]*j:shape_frame[1]*(j+1), :] = frame

        X_test = np.zeros((test_size, shape_frame[0], shape_frame[1]*nb_frame, shape_frame[2]), dtype=int)
        for i, vid in enumerate(vids_test):
            for j, frame in enumerate(vid.frames):
                X_test[i, :, shape_frame[1]*j:shape_frame[1]*(j+1), :] = frame
        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, random_state: int = 1234,
                  epochs: int = 70, batch_size: int = 32, verbose: int = 1):
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
        checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_name, monitor='val_loss', verbose=verbose, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]
        self.model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer='rmsprop',
            metrics=['accuracy']
        )

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,  # données transmises pour une session
            validation_split=0.2,
            callbacks=callbacks_list
        )
        return history

    def evaluate(self, X_in, y_in):
        """
        Evaluate the model

        Parameters
        ----------
        X_in: np.array
            input images
        y_in: np.array
            input labels

        Returns
        -------
        tup: tuple
            tuple of loss and accuracy
        """
        tup = self.model.evaluate(X_in, y_in)
        print("Loss: ", tup[0])
        print("Accuracy: ", tup[1])
        return tup

    def predict(self, seq_img):
        """
        Predict the class of the images

        Parameters
        ----------
        seq_img: np.array 4-dim
            reshaped image or list of images (images avec couleurs)
            Il faut reshape la séquence d'images en (1, 50, 750, 3) pour une vidéo de 15 frames
            Il faut reshape la séquence d'images en (n, 50, 1500, 3) pour n-vidéos de 15 frames
            Si image est en noir et blanc, le shape est (n, 50, 750)

        Returns
        -------
        y_pred: list
            list of predicted classes
        """
        new_seq_img = np.array(seq_img)
        if (len(new_seq_img.shape) == 3):
            new_seq_img = np.reshape(new_seq_img, (seq_img.shape[0], seq_img.shape[1], seq_img.shape[2], 3))
            for i, img in enumerate(seq_img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                new_seq_img[i] = img
        pred = self.model.predict(new_seq_img)
        y_pred = np.argmax(pred, axis=1)
        return y_pred
    
    def predict_label(self, seq_img, y):
        """
        Predict the labels of the images

        Parameters
        ----------
        seq_img: np.array 4-dim
            reshaped image or list of images (images avec couleurs)
            Il faut reshape la séquence d'images en (1, 50, 750, 3) pour une vidéo de 15 frames
            Il faut reshape la séquence d'images en (n, 50, 750, 3) pour n-vidéos de 15 frames
            Si image est en noir et blanc, le shape est (n, 50, 750)

        y: np.array
            all possible output labels (Ex: ['coup droit', 'revers', 'service', 'deplacement'])

        Returns
        -------
        y_pred: list
            list of predicted labels
        """
        new_seq_img = np.array(seq_img)
        if (len(new_seq_img.shape) == 3):
            new_seq_img = np.reshape(new_seq_img, (seq_img.shape[0], seq_img.shape[1], seq_img.shape[2], 3))
            for i, img in enumerate(seq_img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                new_seq_img[i] = img
        pred = self.model.predict(new_seq_img)
        y_pred = np.argmax(pred, axis=1)
        encoder = LabelEncoder()
        encoder.fit(y)
        y_pred = encoder.inverse_transform(y_pred)
        return y_pred