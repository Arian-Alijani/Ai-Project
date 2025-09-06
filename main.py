# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def create_dataset_if_not_exists(filename='data/shoe_size_gender_data.csv', num_samples=100):
    """Generates a sample dataset if it doesn't already exist."""
    if os.path.exists(filename):
        print(f"Dataset '{filename}' already exists.")
        return

    print(f"Generating sample dataset at '{filename}'...")
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Generate synthetic data
    np.random.seed(42)  # for reproducibility
    # Men: larger height, weight, and shoe size
    men_height = np.random.normal(175, 10, num_samples // 2)
    men_weight = np.random.normal(75, 12, num_samples // 2)
    men_shoe_size = np.random.normal(43, 2, num_samples // 2)
    men_gender = ['Male'] * (num_samples // 2)

    # Women: smaller height, weight, and shoe size
    women_height = np.random.normal(162, 8, num_samples // 2)
    women_weight = np.random.normal(60, 10, num_samples // 2)
    women_shoe_size = np.random.normal(38, 1.5, num_samples // 2)
    women_gender = ['Female'] * (num_samples // 2)

    # Combine data
    data = {
        'Height(cm)': np.concatenate([men_height, women_height]),
        'Weight(kg)': np.concatenate([men_weight, women_weight]),
        'Shoe Size(EU)': np.concatenate([men_shoe_size, women_shoe_size]),
        'Gender': np.concatenate([men_gender, women_gender])
    }
    df = pd.DataFrame(data)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    df.to_csv(filename, index=False)
    print("Dataset created successfully.")


def build_and_train_model():
    """Loads data, builds, trains, and evaluates the neural network."""
    # 1. Load and Prepare Data
    data_path = 'data/shoe_size_gender_data.csv'
    create_dataset_if_not_exists(data_path)
    df = pd.read_csv(data_path)

    # Separate features (X) and target (y)
    X = df[['Height(cm)', 'Weight(kg)', 'Shoe Size(EU)']].values
    y = df['Gender'].values

    # Encode target labels (Male/Female to 1/0)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Scale numerical features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Build the Neural Network Model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),  # Regularization to prevent overfitting
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # 3. Compile the Model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 4. Train the Model
    # Early stopping to halt training when validation loss stops improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(X_train_scaled, y_train,
                        epochs=150,
                        batch_size=8,
                        validation_split=0.2, # Use part of training data for validation
                        callbacks=[early_stopping],
                        verbose=1)

    # 5. Evaluate the Model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print("\n--- Model Evaluation ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # 6. Make Predictions on New Data
    print("\n--- New Prediction Example ---")
    # Sample data: Height=180cm, Weight=80kg, Shoe Size=44
    new_data = np.array([[180, 80, 44]])
    new_data_scaled = scaler.transform(new_data) # Use the same scaler

    prediction = model.predict(new_data_scaled)
    predicted_class = (prediction > 0.5).astype("int32")[0][0]
    predicted_gender = label_encoder.inverse_transform([predicted_class])[0]

    print(f"Input Data: Height=180cm, Weight=80kg, Shoe Size=44")
    print(f"Predicted Gender: {predicted_gender} (Raw prediction: {prediction[0][0]:.4f})")

if __name__ == '__main__':
    build_and_train_model()
