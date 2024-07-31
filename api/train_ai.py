import pandas as pd
import numpy as np
import tensorflow as tf
import shap

class AI:
    def preprocess_data(self, df):
        # Ensure 'result' column is present
        if 'result' not in df.columns:
            raise KeyError("'result' column not found in DataFrame")

        # Separate features and target
        X = df.drop(columns=['result'])
        y = df['result']

        # Apply one-hot encoding to features and target
        X_encoded = pd.get_dummies(X, drop_first=False)
        y_encoded = pd.get_dummies(y, drop_first=False)

        # Store feature columns and target columns for later use
        self.feature_columns = X_encoded.columns
        self.target_columns = y_encoded.columns

        # Convert to numpy arrays
        x_values = X_encoded.to_numpy()
        y_values = y_encoded.to_numpy()

        return x_values, y_values

    def preprocess_new_data(self, df):
        for col, encoder in self.encoders.items():
            if col in df.columns:
                unknown_label = 'Unknown'
                df[col] = df[col].apply(lambda x: x if x in encoder.categories_[0] else unknown_label)
                if unknown_label not in encoder.categories_[0]:
                    encoder.categories_[0] = np.append(encoder.categories_[0], unknown_label)
                transformed = encoder.transform(df[[col]])
                df = df.drop(columns=[col])
                for i in range(transformed.shape[1]):
                    df[f"{col}_{i}"] = transformed[:, i]
        return df
    
    def create_nn_model(self, x_train, y_train, num_nodes, epochs, batch_size, lr, dropout_prob):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(10, input_dim=x_train.shape[1], activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_prob))
        
        model.add(tf.keras.layers.Dense(num_nodes, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_prob))
        
        model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        explainer = shap.KernelExplainer(model.predict, x_train)

        # Choose a subset of test data to explain
        subset_idx = np.random.choice(x_train.shape[0], size=len(x_train), replace=False)  # Adjust the size as needed
        x_subset = x_train[subset_idx]

        # Compute SHAP values
        shap_values = explainer.shap_values(x_subset)

        if isinstance(shap_values, list):
            # Multi-class case: average the SHAP values for each class
            shap_values_mean = np.mean([np.abs(class_shap_values).mean(axis=0) for class_shap_values in shap_values], axis=0)
        else:
            shap_values_mean = np.abs(shap_values).mean(axis=0)
            
        means = []
        for i in shap_values_mean:
            means.append(sum(i)/len(i))

        importance = []
        for i in means:
            importance.append(i/sum(means))

        # Create a DataFrame with the mean absolute SHAP values
        shap_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance
        })
        print(shap_df)
        accuracy = history.history['val_accuracy'][-1]
        
        return model, accuracy, shap_df
