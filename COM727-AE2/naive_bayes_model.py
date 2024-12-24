import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTETomek
import joblib

pd.set_option('future.no_silent_downcasting', True)


def load_dataset(file_path):
    print("Loading Dataset...")
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    print("Preprocessing Data...")
    binary_map = {'Yes': 1, 'No': 0}
    data.replace(binary_map, inplace=True)

    
    class_counts = data['Disease'].value_counts()
    valid_classes = class_counts[class_counts > 15].index
    data['Disease'] = data['Disease'].apply(lambda x: x if x in valid_classes else 'Other')
    print(f"Remaining classes after combining: {data['Disease'].nunique()}")
    print("Class distribution:")
    print(data['Disease'].value_counts())

    
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Blood Pressure'] = data['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
    data['Cholesterol Level'] = data['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})

    
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform', subsample=None)

    for col in ['Age']:
        data[col] = discretizer.fit_transform(data[[col]].astype(float))

    return data


def prepare_features_target(data):
    X = data.drop(columns=['Disease', 'Outcome Variable'])
    y = data['Disease']
    return X, y


def train_naive_bayes(X, y):
    print("Splitting Data into Train and Test Sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Balancing Training Data...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)

    print("Training Multinomial Naive Bayes Classifier...")
    model = MultinomialNB(alpha=0.01)
    model.fit(X_train_res, y_train_res)

    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    
    joblib.dump(model, "final_naive_bayes_model.pkl")
    print("Final Naive Bayes model saved as 'final_naive_bayes_model.pkl'")

    return model


def load_condition_guidance():
    print("Loading Condition Guidance...")
    return pd.read_csv("condition_guidance.csv")


def interactive_conversation(predicted_condition):
    guidance_df = load_condition_guidance()
    condition_guidance = guidance_df[guidance_df['Condition'] == predicted_condition]

    print(f"\nHere is more information about '{predicted_condition}':")
    for idx, row in condition_guidance.iterrows():
        print(f"\n{row['Question']}")
        print(f"Answer: {row['Answer']}")
        user_input = input("\nWould you like to know more? (Yes/No): ").lower()
        if user_input == 'no':
            print("Returning to main menu...\n")
            return
    print("\nYou have explored all information for this condition. Returning to main menu...\n")


def main():
    file_path = "Disease_symptom_and_patient_profile_dataset.csv"
    data = load_dataset(file_path)
    data = preprocess_data(data)
    X, y = prepare_features_target(data)
    model = train_naive_bayes(X, y)

    print("\nInteractive Symptom Checker:")
    while True:
        print("\nPlease answer the following questions (or type 'X' to return to the main menu):")
        age_input = input("Enter your age: ")
        if age_input.lower() == 'x':
            print("Returning to the main menu...\n")
            break

        try:
            age = float(age_input)
            gender = input("Enter your gender (Male/Female): ").strip().capitalize()
            if gender.lower() == 'x':
                print("Returning to the main menu...\n")
                break

            fever = input("Do you have Fever? (Yes/No): ").strip().capitalize()
            if fever.lower() == 'x':
                print("Returning to the main menu...\n")
                break

            cough = input("Do you have Cough? (Yes/No): ").strip().capitalize()
            if cough.lower() == 'x':
                print("Returning to the main menu...\n")
                break

            fatigue = input("Do you feel Fatigue? (Yes/No): ").strip().capitalize()
            if fatigue.lower() == 'x':
                print("Returning to the main menu...\n")
                break

            
            input_data = pd.DataFrame([[fever, cough, fatigue, age, gender, 1, 1]],
                                    columns=['Fever', 'Cough', 'Fatigue', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level'])
            input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0})
            input_data.replace({'Yes': 1, 'No': 0}, inplace=True)

            
            predicted_condition = model.predict(input_data)[0]
            print(f"\nBased on your inputs, the predicted condition is: **{predicted_condition}**")

            
            user_input = input("Would you like to know more about this condition? (Yes/No): ").lower()
            if user_input == 'yes':
                interactive_conversation(predicted_condition)
            else:
                print("Returning to the symptom checker...\n")

        except ValueError:
            print("Invalid input! Please enter valid values.")


if __name__ == "__main__":
    main()
