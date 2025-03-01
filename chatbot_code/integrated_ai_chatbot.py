import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, LSTM, Embedding
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.semi_supervised import LabelPropagation
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report

from scipy.sparse import hstack


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.utils import shuffle


from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def load_condition_guidance():
    print("Loading Condition Guidance...")
    return pd.read_csv("condition_guidance.csv")


def interactive_rnn_conversation(predicted_condition):
    print("Loading Condition Guidance...")
    guidance_df = load_condition_guidance()

    
    condition_guidance = guidance_df[guidance_df['Condition'] == predicted_condition]

    
    if condition_guidance.empty:
        print(f"Sorry, no additional information is available for '{predicted_condition}'.")
        print("Consider consulting a healthcare professional for more details.")
        return

    
    print(f"\nHere is more information about '{predicted_condition}':")
    while True:
        for idx, row in condition_guidance.iterrows():
            print(f"\n{row['Question']}")
            print(f"Answer: {row['Answer']}")
            user_input = input("\nWould you like to know more? (Yes/No/X to return): ").strip().lower()
            if user_input == 'no':
                print("Alright! Let me know if you need anything else.")
                return
            elif user_input == 'x':
                print("Returning to the main menu...\n")
                return
        print("\nYou have explored all available information for this condition.")
        print("Returning to the main menu...\n")
        return




def load_dataset(file_path):
    print("Loading Dataset...")
    data = pd.read_csv(file_path)
    return data



def preprocess_data(data):
    print("Preprocessing Data...")
    binary_map = {'Yes': 1, 'No': 0}
    pd.set_option('future.no_silent_downcasting', True)
    data.replace(binary_map, inplace=True)

    
    class_counts = data['Disease'].value_counts()
    valid_classes = class_counts[class_counts > 10].index
    data['Disease'] = data['Disease'].apply(lambda x: x if x in valid_classes else 'Other')
    print(f"Remaining classes after combining: {data['Disease'].nunique()}")

    
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Blood Pressure'] = data['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
    data['Cholesterol Level'] = data['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})

    return data



def ensure_numeric(X):
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X

def run_deep_learning_model():
    print("Running Deep Learning Health Prediction...")
    file_path = "Disease_symptom_and_patient_profile_dataset.csv"

    
    data = load_dataset(file_path)
    data = preprocess_data(data)
    X, y = prepare_features_target(data)
    X = ensure_numeric(X)

    
    X_resampled, y_resampled = balance_classes(X, y)

    
    X_scaled, scaler = scale_features(X_resampled)

    
    rnn_health_prediction(X_scaled, y_resampled)


def rnn_health_prediction(X_scaled, y):
    print("Running Deep Learning Health Prediction (RNN)...")

   
    y_encoded = pd.get_dummies(y).values

    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    
    X_train_reshaped = X_train.to_numpy().reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.to_numpy().reshape((X_test.shape[0], 1, X_test.shape[1]))

    
    model = Sequential([
        Input(shape=(1, X_train.shape[1])),
        LSTM(64, activation='tanh', return_sequences=False),
        Dense(32, activation='relu'),
        Dense(y_encoded.shape[1], activation='softmax')  
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=16, verbose=1)

    
    loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"RNN Model Accuracy: {accuracy:.2f}")

    
    model.save("final_rnn_model.h5")
    print("Final RNN model saved as 'final_rnn_model.h5'")

    
    print("\nInteractive Health Prediction:")
    sample_input = X_test_reshaped[:1]
    predicted_class = model.predict(sample_input).argmax(axis=1)[0]
    predicted_condition = pd.get_dummies(y).columns[predicted_class]
    print(f"Based on the analysis, the predicted condition is: **{predicted_condition}**\n")

    
    while True:
        user_input = input(f"Would you like to know more about '{predicted_condition}'? (Yes/No/X to return): ").lower()
        if user_input == 'yes':
            interactive_rnn_conversation(predicted_condition)

        elif user_input == 'x':
            print("Returning to the main menu...")
            return
        else:
            print("Alright! Let me know if you need anything else.")



from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def dbscan_clustering(X):
    print("Running DBSCAN Clustering...")

    
    X = ensure_numeric(X)

   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    
    dbscan = DBSCAN(eps=1.0, min_samples=3)
    clusters = dbscan.fit_predict(X_pca)

    
    if len(set(clusters)) > 1:  
        silhouette = silhouette_score(X_pca, clusters)
        print(f"Silhouette Score: {silhouette:.2f}")
    else:
        print("Silhouette Score cannot be calculated (only one cluster detected).")

    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
    plt.title("DBSCAN Clustering after PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster')
    plt.show()


def hierarchical_clustering(X):
    print("Running Hierarchical Clustering...")

    
    X_numeric = X.select_dtypes(include=['number'])

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    
    clustering = AgglomerativeClustering(n_clusters=3)
    clusters = clustering.fit_predict(X_scaled)

    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title("Hierarchical Clustering Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

    print("Hierarchical Clustering completed successfully!")


from sklearn.preprocessing import StandardScaler


def semi_supervised_learning(X, y):
    print("Running Semi-Supervised Learning...")

    
    X_numeric = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    
    y_encoded[:20] = -1

    
    from sklearn.semi_supervised import LabelPropagation
    model = LabelPropagation()
    model.fit(X_scaled, y_encoded)

    
    predictions = model.predict(X_scaled)
    print("Semi-Supervised Learning Complete.")
    print(f"Predictions: {label_encoder.inverse_transform(predictions)}")



vectorizer = TfidfVectorizer()



def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    
    words = nltk.word_tokenize(text.lower())
    clean_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return " ".join(clean_words)


vectorizer = TfidfVectorizer()


def train_nlp_model(file_path):
    print("Training NLP Symptom Classifier using Disease Dataset...")

   
    data = pd.read_csv(file_path)

    
    if 'Symptom Description' not in data.columns or 'Disease' not in data.columns:
        print("Error: Dataset must contain 'Symptom Description' and 'Disease' columns.")
        return

    
    print("Preprocessing symptom descriptions...")
    data['processed_symptoms'] = data['Symptom Description'].apply(preprocess_text)

    
    print("Vectorizing symptom descriptions...")
    X = vectorizer.fit_transform(data['processed_symptoms'])
    y = data['Disease']

    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    print("Training Naive Bayes Classifier...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    
    joblib.dump(model, "nlp_symptom_model.pkl")
    joblib.dump(vectorizer, "nlp_vectorizer.pkl")
    print("NLP Symptom Classifier Trained and Saved Successfully.")



def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text.lower())
    clean_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(clean_words)



from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib



def train_nlp_model():
    print("\nTraining NLP Model for Symptom Classification...")

    
    file_path = "Disease_symptom_and_patient_profile_dataset.csv"
    data = pd.read_csv(file_path)
    print("Dataset Columns:", data.columns)

    
    print("Generating symptom descriptions...")
    data['Symptom_Description'] = data.apply(
        lambda row: f"{row['Fever']} {row['Cough']} {row['Fatigue']} {row['Difficulty Breathing']} {row['Age']} {row['Gender']}", axis=1
    )

    
    print("Combining rare classes into 'Other'...")
    min_class_count = 5  
    data['Disease'] = data['Disease'].apply(lambda x: x if data['Disease'].value_counts()[x] >= min_class_count else 'Other')
    print("Class Distribution After Combining Rare Classes:")
    print(data['Disease'].value_counts())

    
    X = data['Symptom_Description']
    y = data['Disease']

    
    print("Vectorizing text data using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=500)
    X_vectorized = vectorizer.fit_transform(X)

    
    print("Balancing classes using SMOTE...")
    smote = SMOTE(k_neighbors=3, random_state=42)  
    X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)

    
    print("Training NLP model with GridSearchCV for hyperparameter tuning...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    model = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000), param_grid, cv=3)
    model.fit(X_resampled, y_resampled)

    
    print("Evaluating the model...")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete. Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    
    print("Saving the model and vectorizer...")
    joblib.dump(model, "final_nlp_model.pkl")
    joblib.dump(vectorizer, "final_tfidf_vectorizer.pkl")
    print("NLP Model and Vectorizer saved successfully.")

import joblib
import pandas as pd


def nlp_user_symptoms():
    """
    Predict the condition based on user-provided symptoms.
    """
    try:
        print("\nRunning NLP for User Symptoms...")

        
        print("\nPlease answer the following questions:")
        fever = input("Do you have Fever? (Yes/No): ").strip().capitalize()
        cough = input("Do you have Cough? (Yes/No): ").strip().capitalize()
        fatigue = input("Do you feel Fatigue? (Yes/No): ").strip().capitalize()
        breathing = input("Do you have Difficulty Breathing? (Yes/No): ").strip().capitalize()
        age = input("Enter your Age: ").strip()
        gender = input("Enter your Gender (Male/Female): ").strip().capitalize()

        
        input_description = (f"Fever: {fever}, Cough: {cough}, Fatigue: {fatigue}, "
                             f"Difficulty Breathing: {breathing}, Age: {age}, Gender: {gender}")

        
        model = joblib.load("nlp_model.pkl")
        vectorizer = joblib.load("nlp_vectorizer.pkl")

        
        input_vectorized = vectorizer.transform([input_description])

        
        predicted_condition = model.predict(input_vectorized)[0]
        print(f"\nBased on your symptoms, the predicted condition is: **{predicted_condition}**")

       
        more_info = input(f"Would you like to know more about '{predicted_condition}'? (Yes/No/X to return): ").strip().lower()
        if more_info == 'yes':
            interactive_nlp_conversation(predicted_condition)  
        elif more_info == 'x':
            print("Returning to the main menu...\n")
        else:
            print("Alright! Let me know if you need anything else.\n")

    except FileNotFoundError:
        print("Error: Trained NLP model not found. Please train the model first using option 22.")
    except Exception as e:
        print(f"Error during prediction: {e}")


def interactive_nlp_conversation(predicted_condition):
    """
    Load condition guidance and provide detailed information.
    """
    try:
        print("\nLoading Condition Guidance...")
        guidance_df = pd.read_csv("condition_guidance.csv")
        condition_guidance = guidance_df[guidance_df['Condition'] == predicted_condition]

        if condition_guidance.empty:
            print(f"Sorry, no detailed information is available for '{predicted_condition}'.")
            print("Consider consulting a healthcare professional for more details.")
        else:
            print(f"\nHere is more information about '{predicted_condition}':")
            for _, row in condition_guidance.iterrows():
                print(f"\nQ: {row['Question']}")
                print(f"A: {row['Answer']}")

    except FileNotFoundError:
        print("Error: 'condition_guidance.csv' not found. Ensure the file is in the correct location.")
    except Exception as e:
        print(f"Error while loading condition guidance: {e}")


import pandas as pd


def interactive_rnn_conversation(predicted_condition):
    print("Loading Condition Guidance...")
    try:
        
        guidance_df = pd.read_csv("condition_guidance.csv")
    except FileNotFoundError:
        print("Error: condition_guidance.csv file not found. Please ensure the file exists.")
        return

    if predicted_condition in guidance_df['Condition'].unique():
        print(f"\nHere is more information about '{predicted_condition}':\n")

        
        condition_guidance = guidance_df[guidance_df['Condition'] == predicted_condition]

        
        for _, row in condition_guidance.iterrows():
            print(f"Q: {row['Question']}")
            print(f"A: {row['Answer']}\n")

           
            user_input = input(
                "Would you like to continue exploring this condition? (Yes/No/X to return): ").strip().lower()
            if user_input == 'no':
                print("Alright! Let me know if you need anything else.\n")
                return
            elif user_input == 'x':
                print("Returning to the main menu...\n")
                return
        print("You have explored all available information for this condition.\n")
    else:
        
        print(f"\nSorry, detailed information for '{predicted_condition}' is unavailable in the current database.\n")
        print("Here are some general health tips:\n")
        print("- Maintain a balanced diet and exercise regularly.")
        print("- Consult a healthcare professional for proper diagnosis.")
        print("- Monitor symptoms closely and seek immediate care if needed.")
        print("Returning to the main menu...\n")



def sentiment_analysis():
    print("Running Sentiment Analysis...")
    user_input = input("Enter your thoughts or feedback: ")
    sentiment = TextBlob(user_input).sentiment
    print(f"Polarity: {sentiment.polarity:.2f} | Subjectivity: {sentiment.subjectivity:.2f}")
    print("Sentiment: Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral")



def prepare_features_target(data):
    X = data.drop(columns=['Disease'])
    y = data['Disease']
    return X, y



def balance_classes(X, y):
    print("Balancing Classes with SMOTETomek...")
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    return X_resampled, y_resampled
