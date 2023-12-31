
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, SimpleRNN

class MachineLearningOperations:
    def __init__(self, excel_path, x, label_column):
        self.excel_path = excel_path
        self.x = x
        self.label_column = label_column
        self.df = pd.read_excel(excel_path)

    def knn(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        knn_model = KNeighborsClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            knn_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = knn_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def random_forest(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        rf_model = RandomForestClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            rf_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = rf_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def logistic_regression(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        lr_model = LogisticRegression()

        for epoch in range(1, e+1):
            start_time = time.time()
            lr_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = lr_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def decision_tree(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        dt_model = DecisionTreeClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            dt_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = dt_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def linear_regression(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        lr_model = LinearRegression()

        start_time = time.time()
        lr_model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time

        y_pred = lr_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f'Time: {elapsed_time:.2f}s - Mean Squared Error: {mse}')

        return y_test, y_pred, mse

    def gru(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        # Assuming numerical_features is a 3D array [samples, timesteps, features]
        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        model = Sequential()
        model.add(GRU(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for epoch in range(1, e+1):
            start_time = time.time()
            model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            elapsed_time = end_time - start_time

            _, accuracy = model.evaluate(X_test, y_test)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        y_pred = model.predict_classes(X_test)

        return y_test, y_pred, accuracy

    def rnn(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        # Assuming numerical_features is a 3D array [samples, timesteps, features]
        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        model = Sequential()
        model.add(SimpleRNN(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for epoch in range(1, e+1):
            start_time = time.time()
            model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            elapsed_time = end_time - start_time

            _, accuracy = model.evaluate(X_test, y_test)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        y_pred = model.predict_classes(X_test)

        return y_test, y_pred, accuracy

    def lstm(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        # Assuming numerical_features is a 3D array [samples, timesteps, features]
        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        model = Sequential()
        model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for epoch in range(1, e+1):
            start_time = time.time()
            model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            elapsed_time = end_time - start_time

            _, accuracy = model.evaluate(X_test, y_test)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        y_pred = model.predict_classes(X_test)

        return y_test, y_pred, accuracy

    def svm(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        svm_model = SVC()

        for epoch in range(1, e+1):
            start_time = time.time()
            svm_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = svm_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def naive_bayes(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        nb_model = GaussianNB()

        start_time = time.time()
        nb_model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time

        y_pred = nb_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def adaboost(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        adaboost_model = AdaBoostClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            adaboost_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = adaboost_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def bagging(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        bagging_model = BaggingClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            bagging_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = bagging_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy
# ... (previous code)

    def gru(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        # Assuming numerical_features is a 3D array [samples, timesteps, features]
        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        model = Sequential()
        model.add(GRU(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for epoch in range(1, e+1):
            start_time = time.time()
            model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            elapsed_time = end_time - start_time

            _, accuracy = model.evaluate(X_test, y_test)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        y_pred = model.predict_classes(X_test)

        return y_test, y_pred, accuracy

    def neural_network(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        model = Sequential()
        model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for epoch in range(1, e+1):
            start_time = time.time()
            model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            elapsed_time = end_time - start_time

            _, accuracy = model.evaluate(X_test, y_test)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        y_pred = model.predict_classes(X_test)

        return y_test, y_pred, accuracy

    def decision_tree(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        dt_model = DecisionTreeClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            dt_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = dt_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def rnn(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        # Assuming numerical_features is a 3D array [samples, timesteps, features]
        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        model = Sequential()
        model.add(SimpleRNN(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for epoch in range(1, e+1):
            start_time = time.time()
            model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            elapsed_time = end_time - start_time

            _, accuracy = model.evaluate(X_test, y_test)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        y_pred = model.predict_classes(X_test)

        return y_test, y_pred, accuracy

    def lstm(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        # Assuming numerical_features is a 3D array [samples, timesteps, features]
        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        model = Sequential()
        model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for epoch in range(1, e+1):
            start_time = time.time()
            model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            elapsed_time = end_time - start_time

            _, accuracy = model.evaluate(X_test, y_test)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        y_pred = model.predict_classes(X_test)

        return y_test, y_pred, accuracy

    def svm(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        svm_model = SVC()

        for epoch in range(1, e+1):
            start_time = time.time()
            svm_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = svm_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def random_forest(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        rf_model = RandomForestClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            rf_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = rf_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def logistic_regression(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        lr_model = LogisticRegression()

        for epoch in range(1, e+1):
            start_time = time.time()
            lr_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = lr_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def linear_regression(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        lr_model = LinearRegression()

        start_time = time.time()
        lr_model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time

        y_pred = lr_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f'Time: {elapsed_time:.2f}s - Mean Squared Error: {mse}')

        return y_test, y_pred, mse

    # Add the remaining algorithms if they don't exist in the class
    # (SVM, Random Forest, Logistic Regression)...
    
# ... (continue with the remaining code)
    def neural_network(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        model = Sequential()
        model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        for epoch in range(1, e+1):
            start_time = time.time()
            model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            end_time = time.time()
            elapsed_time = end_time - start_time

            _, accuracy = model.evaluate(X_test, y_test)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        y_pred = model.predict_classes(X_test)

        return y_test, y_pred, accuracy

    def svm(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        svm_model = SVC()

        for epoch in range(1, e+1):
            start_time = time.time()
            svm_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = svm_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def naive_bayes(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        nb_model = GaussianNB()

        start_time = time.time()
        nb_model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time

        y_pred = nb_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def adaboost(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        adaboost_model = AdaBoostClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            adaboost_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = adaboost_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def bagging(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        bagging_model = BaggingClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            bagging_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = bagging_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def extra_trees(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        extra_trees_model = ExtraTreesClassifier()

        for epoch in range(1, e+1):
            start_time = time.time()
            extra_trees_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = extra_trees_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Test Accuracy: {accuracy}')

        return y_test, y_pred, accuracy

    def mlp_regressor(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        mlp_regressor_model = MLPRegressor()

        for epoch in range(1, e+1):
            start_time = time.time()
            mlp_regressor_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = mlp_regressor_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Mean Squared Error: {mse}')

        return y_test, y_pred, mse

    def decision_tree_regressor(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        dt_regressor_model = DecisionTreeRegressor()

        for epoch in range(1, e+1):
            start_time = time.time()
            dt_regressor_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = dt_regressor_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Mean Squared Error: {mse}')

        return y_test, y_pred, mse

    def random_forest_regressor(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        rf_regressor_model = RandomForestRegressor()

        for epoch in range(1, e+1):
            start_time = time.time()
            rf_regressor_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = rf_regressor_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Mean Squared Error: {mse}')

        return y_test, y_pred, mse

    def svm_regressor(self, e):
        numerical_features = self.df[self.x].values
        labels = self.df[self.label_column].values

        numerical_features_standardized = StandardScaler().fit_transform(numerical_features)

        X_train, X_test, y_train, y_test = train_test_split(numerical_features_standardized, labels, test_size=0.3, random_state=1)

        svm_regressor_model = SVR()

        for epoch in range(1, e+1):
            start_time = time.time()
            svm_regressor_model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = svm_regressor_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            print(f'Epoch {epoch} - Time: {elapsed_time:.2f}s - Mean Squared Error: {mse}')

        return y_test, y_pred, mse

