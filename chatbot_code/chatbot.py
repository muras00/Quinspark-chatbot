
# package version control for PyCharm:
# Python == 3.10 (set in environment/interpreter)
# NumPy == latest (2.1.3 or 1.26.4)
# NLTK == 3.9.1
# TensorFlow == 2.14.1
# Keras == 2.14.0 (comes installed with TensorFlow)
# Googletrans == 3.1.0a0
# Other libraries == latest

# Upload the following files to the same project file:
# chatbot_emergency_map.py, naive_bayes_model.py, integrated_ai_chatbot.py
# medications.json, symptom_illness_dataset.csv, mental_health_tips_extended.csv,
# symptom_checker_model.pkl, best_enhanced_symptom_model.pkl, model_comparison_results.pkl
# scaler.pkl, best_models.pkl, label_encoder_y.pkl, intents.json, health_tips.json,
# instructions_words.pkl, instructions_classes.pkl, ANN_instructions_model.keras,
# clustering_analysis.py, Disease_symptom_and_patient_profile_dataset.csv,
# time_series_forecasting.py, time_series_symptoms.csv, condition_guidance.csv
# nlp_model.pkl, nlp_vectorizer.pkl

import random
import json
import pickle
import numpy as np
import nltk
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import chatbot_emergency_map # Upload the file "chatbot_emergency_map.py" under "content" folder
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from googletrans import Translator
import webbrowser
import naive_bayes_model  # Upload the file "naive_bayes_model.py" under "content" folder
#import nltk
import sys

from integrated_ai_chatbot import ( # Upload the file "integrated_ai_chatbot.py" under "content" folder
    load_dataset,
    run_deep_learning_model,
    preprocess_data,
    prepare_features_target,
    ensure_numeric,
    rnn_health_prediction,
    dbscan_clustering,
    hierarchical_clustering,
    semi_supervised_learning,
    nlp_user_symptoms,
    sentiment_analysis
)
#import nltk
#from googletrans import Translator

from integrated_ai_chatbot import train_nlp_model


def symptom_checker_naive_bayes():
    try:
        print("\nRunning Symptom Checker using Naive Bayes...")
        naive_bayes_model.run_naive_bayes()  
    except Exception as e:
        print(f"An error occurred while running Naive Bayes: {e}")

medications = json.loads(open('medications.json').read())
symptom_data = pd.read_csv("symptom_illness_dataset.csv")
mental_health_tips = pd.read_csv("mental_health_tips_extended.csv")
model = joblib.load("symptom_checker_model.pkl")


print("Loading the trained symptom prediction model...")
model = joblib.load("best_enhanced_symptom_model.pkl")
comparison_results = joblib.load("model_comparison_results.pkl")  

scaler = joblib.load("scaler.pkl")
best_models = joblib.load("best_models.pkl")
label_encoder_y = joblib.load("label_encoder_y.pkl")  

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
health_tips = json.loads(open('health_tips.json').read())

instructions_words = pickle.load(open('instructions_words.pkl', 'rb'))
instructions_classes = pickle.load(open('instructions_classes.pkl', 'rb'))
instructions_model = load_model('ANN_instructions_model.keras')

translator = Translator()
lang = "en"

language_list = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn',
                  'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-cn', 'chinese (traditional)': 'zh-tw',
                  'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl',
                  'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'gujarati': 'gu', 'haitian creole': 'ht',
                  'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'he',  'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'indonesian': 'id',
                  'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'korean': 'ko', 'kurdish': 'ku', 'kurmanji': 'ku',
                  'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lithuanian': 'lt', 'luxembourgish': 'lb', 'macedonian': 'mk', 'malagasy': 'mg', 'malay': 'ms',
                  'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'mongolian': 'mn', 'myanmar': 'my', 'burmese': 'my', 'nepali': 'ne', 'norwegian': 'no',
                  'odia': 'or', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm',
                  'scots gaelic': 'gd', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so',
                  'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr',
                  'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(instructions_words)
    for w in sentence_words:
        for i, word in enumerate(instructions_words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence) #bow: Bag Of Words, feed the data into the neural network
    res = instructions_model.predict(np.array([bow]), verbose=0)[0] #res: result. [0] as index 0
    # Senju - Notes: added verbose=0 to hide the predict progress bar
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': instructions_classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result, tag

def display_menu():
    print(translator.translate("""Please select one of the following options:
[1] Symptom checker - N/A
[2] First aid instructions
[3] Display emergency contacts
[4] Display health tips
[5] Display resource links
[6] Language support
[7] Display emergency services on map
[8] Find medications for a condition
[9] Symptom checker (AI-based prediction)
[10] Mental health support
[11] Display model comparison results
[12] Dynamic model selection for predictions
[13] Analyze patient clusters using K-Means
[14] Symptom checker using Naive Bayes
[15] Forecast health trends using Time Series
[16] Deep Learning Health Prediction
[17] DBSCAN Clustering
[18] Hierarchical Clustering
[19] Semi-Supervised Learning
[20] NLP for User Symptoms
[21] Sentiment Analysis
[22] Train NLP
[X] Exit application

""", dest=lang).text, "\n")
    selected_option = input(translator.translate("You: ", dest=lang).text)
    return selected_option

print("QuinSpark Chatbot is here!\n")

def run_deep_learning_model():
    file_path = "Disease_symptom_and_patient_profile_dataset.csv"
    data = load_dataset(file_path)
    data = preprocess_data(data)
    X, y = prepare_features_target(data)
    X = ensure_numeric(X)
    rnn_health_prediction(X, y)

def display_model_comparison():
    print("\nModel Comparison Results:")
    print("Model Performance Metrics (Accuracy and ROC-AUC):\n")
    for model_name, metrics in comparison_results.items():
        print(f"{model_name}: Accuracy = {metrics['Accuracy']:.2f}, ROC-AUC = {metrics['ROC-AUC']:.2f}")

    
    while True:
        return_choice = input("Enter 'X' to return to the main menu: ").strip().lower()
        if return_choice == 'x':
            break
        else:
            print("Invalid input. Please enter 'X' to return to the main menu.")



def dynamic_model_selection():
    print("\nWelcome to the Dynamic Symptom Checker!")
    print("Please select a model for predictions:")

    
    for idx, model_name in enumerate(best_models.keys(), 1):
        print(f"[{idx}] {model_name}")

    
    try:
        model_choice = int(input("Enter the number corresponding to your model choice: ").strip())
        model_names = list(best_models.keys())
        selected_model_name = model_names[model_choice - 1]
        model = best_models[selected_model_name]
        print(f"You have selected: {selected_model_name}\n")
    except (ValueError, IndexError):
        print("Invalid input. Defaulting to Random Forest.")
        model = best_models["Random Forest"]

    
    while True:
        try:
            print("\nPlease answer the following questions with 'Yes' or 'No'.")
            fever = input("Do you have Fever? (Yes/No): ").strip().lower()
            cough = input("Do you have Cough? (Yes/No): ").strip().lower()
            fatigue = input("Do you have Fatigue? (Yes/No): ").strip().lower()
            breathing = input("Do you have Difficulty Breathing? (Yes/No): ").strip().lower()
            age = int(input("Enter your age (e.g., 25): ").strip())
            gender = input("Enter your gender (Male/Female): ").strip().lower()
            blood_pressure = input("Blood Pressure (Low/Normal/High): ").strip().lower()
            cholesterol = input("Cholesterol Level (Low/Normal/High): ").strip().lower()

            
            symptom_map = {"yes": 1, "no": 0}
            gender_map = {"male": 1, "female": 0}
            bp_map = {"low": 0, "normal": 1, "high": 2}
            cholesterol_map = {"low": 0, "normal": 1, "high": 2}

            user_input = {
                'Fever': symptom_map.get(fever, 0),
                'Cough': symptom_map.get(cough, 0),
                'Fatigue': symptom_map.get(fatigue, 0),
                'Difficulty Breathing': symptom_map.get(breathing, 0),
                'Age': age,
                'Gender': gender_map.get(gender, 0),
                'Blood Pressure': bp_map.get(blood_pressure, 1),
                'Cholesterol Level': cholesterol_map.get(cholesterol, 1)
            }

            
            user_input_df = pd.DataFrame([user_input])
            user_input_scaled = scaler.transform(user_input_df)

            
            prediction_proba = model.predict_proba(user_input_scaled)
            predicted_label = np.argmax(prediction_proba[0])
            confidence = prediction_proba[0][predicted_label]
            predicted_disease = label_encoder_y.inverse_transform([predicted_label])[0]

            
            print(f"\nPredicted Illness: {predicted_disease}")
            print(f"Prediction Confidence: {confidence * 100:.2f}%")
            print("Note: This is an AI-based prediction. Please consult a healthcare professional.\n")

            
            exit_choice = input("Enter 'X' to return to the main menu, or press Enter to check again: ").strip().lower()
            if exit_choice == 'x':  
                break

        except Exception as e:
            print(f"Error: {e}. Please try again.\n")

def analyze_clusters():
    print("\nPerforming Clustering Analysis...")
    exec(open("clustering_analysis.py").read())

def symptom_checker_naive_bayes():
    print("\nRunning Symptom Checker using Naive Bayes...")
    from naive_bayes_model import main as run_naive_bayes

    run_naive_bayes()

def forecast_health_trends():
    print("\nForecasting Health Trends...")
    from time_series_forecasting import forecast_health_trends_interactive
    forecast_health_trends_interactive()


# Main - loop starts here
while True:
    res = ""
    tag = ""

    # Display menu
    selected_option = display_menu()

    if selected_option == '1':
      print("\nComing soon...\n")

    # First-aid instructions
    elif selected_option == '2':
        print("\n", translator.translate("I can provide you with first aid instructions, "
              "how can I help you? (Enter \"X\" to go back to menu)", dest=lang).text, "\n")
        while True:
          message = input(translator.translate("You: ", dest=lang).text)
          if message == 'X':
            print("\n", translator.translate("Going back to menu...", dest=lang).text, "\n")
            break
          else:
            message = translator.translate(message, dest="en").text
            ints = predict_class(message)
            if ints:  # Senju - Notes: added if-else for error-handling
                res, tag = get_response(ints, intents)
                tag = translator.translate("Instruction for " + tag, dest=lang).text
                res = translator.translate(res, dest=lang).text
                res = "\033[1m" + tag + "\033[0m\n" + res + "\n\n" + translator.translate("Can I help you with anything else?", dest=lang).text
            else:
                res = translator.translate("Sorry, I didn't get that.", dest=lang).text + "\n" + translator.translate("Can I help you with anything else?", dest=lang).text
            print(f"\n{res}\n")

    # Display emergency contacts
    elif selected_option == '3':
      while True:
        header = "\n\033[1m" + translator.translate("Emergency & First Aid Hotline Numbers:", dest=lang).text + "\033[0m"
        body = "\n\n" + translator.translate("""In case of an emergency or if you need immediate medical assistance, here are the important numbers you should know:

Emergency (Police, Fire, Ambulance):
📞 911 (USA), 112 (Europe & International)

Poison Control:
📞 1-800-222-1222 (USA)
📞 084-524-624 (UK)
📞 13 11 26 (Australia)

Emergency Medical Services (EMS):
📞 112 (Most countries)
📞 999 (UK)

First Aid (General):
📞 Red Cross First Aid:
📞 1-800-733-2767 (USA)
📞 0300 33 33 105 (UK)

Mental Health Crisis Hotline:
📞 988 (USA)
📞 116 123 (UK)

Suicide Prevention Hotline:
📞 1-800-273-8255 (USA)
📞 0800 012 033 (New Zealand)

Enter \"X\" to go back to menu

""", dest=lang).text + "\n"

        print(header, body)
        message = input(translator.translate("You: ", dest=lang).text)
        if message == 'X':
          print("\n", translator.translate("Going back to menu...", dest=lang).text, "\n")
          break
        else:
          continue

    # Display health tips
    elif selected_option == '4':
      while True:
        rnd = str(random.randint(1, 10))
        for i in health_tips['intents']:
          if rnd in i['patterns']:
              res1 = translator.translate(i['responses'][0], dest=lang).text
              res2 = translator.translate(i['responses'][1], dest=lang).text
              res3 = translator.translate(i['responses'][2], dest=lang).text
              res4 = translator.translate(i['responses'][3], dest=lang).text
              res = "\033[1m\033[3m" + res1 + "\033[0m" + res2 + "\n\n" + "\033[1m\033[3m" + res3 + "\033[0m" + res4
              tag = translator.translate(i['tag'][0], dest=lang).text
              break
        generate_another = translator.translate("Should I generate another response?", dest=lang).text
        res = "\033[1m\033[4m" + tag + "\n\033[0m\n" + res + "\n\n\n\n" + generate_another + " (Y/N)"
        print(f"\n{res}\n")

        message = input(translator.translate("You: ", dest=lang).text)
        if message == 'Y':
          print("\n\n")
          continue
        else:
          print("\n", translator.translate("Going back to menu...", dest=lang).text, "\n")
          break



    # Display resource links
    elif selected_option == '5':
      while True:
        print(translator.translate("""\n
1. NHS
https://www.nhs.uk/

2. Mayo Clinic
https://www.mayoclinic.org/

3. CDC (Centers for Disease Control and Prevention)
https://www.cdc.gov/

4. WHO (World Health Organization)
https://www.who.int/health-topics

5. MedlinePlus
https://medlineplus.gov/

6. NIH (National Institutes of Health)
https://www.nih.gov/health-information

7. Local Health Services (UK)
https://www.nhs.uk/nhs-services/services-near-you/

Enter \"X\" to go back to menu

""", dest=lang).text, "\n")

        message = input(translator.translate("You: ", dest=lang).text)
        if message == 'X':
          print("\n", translator.translate("Going back to menu...", dest=lang).text, "\n")
          break
        else:
          continue

    # Language support
    elif selected_option == '6':
      while True:
        selected_language = "\n" + translator.translate("Select your preferred language", dest=lang).text + "\n(" + translator.translate("Enter \"X\" to go back to menu", dest=lang).text + "): "
        selected_language = input(selected_language)

        if selected_language == 'X':
          print("\n", translator.translate("Going back to menu...", dest=lang).text, "\n")
          break

        selected_language = translator.translate(selected_language, dest="en").text.lower()

        if selected_language == "chinese":
          selected_chinese = translator.translate("Simplified or traditional chinese?", dest=lang).text
          selected_chinese = input(selected_chinese + " (\"simplified\"/\"traditional\") ").lower()
          if selected_chinese == "simplified":
            lang = "zh-cn"
          elif selected_chinese == "traditional":
            lang = "zh-tw"
          selected_language = translator.translate("Your language is set to "+ selected_language, dest=lang).text
          print(f"{selected_language} ({lang})\n")
          break

        elif selected_language in language_list.keys():
          lang = language_list.get(selected_language)
          print(f"Your language is set to {selected_language} ({lang})\n")
          break

        else:
          print("\n", translator.translate("Please enter a valid language name", dest=lang).text, "\n")
          continue

    
    elif selected_option == '7':
        location_name = input(translator.translate("Enter the location (e.g., London): ", dest=lang).text).strip()
        if not location_name:
            print(translator.translate("Invalid input. Please try again.", dest=lang).text)
            continue

        print(translator.translate("Generating map for emergency services, please wait...", dest=lang).text)
        map_image_path = chatbot_emergency_map.load_emergency_services_map(location_name)

        
        if "Sorry" in map_image_path:
            print(translator.translate(map_image_path, dest=lang).text)
        else:
            print(translator.translate(f"Here is the map of emergency services in {location_name}.", dest=lang).text)
            print(f"📍 Opening the map...")

            
            if map_image_path.endswith(".png"):
                os.startfile(map_image_path)  
            elif map_image_path.endswith(".html"):
                webbrowser.open_new_tab(f"file://{os.path.abspath(map_image_path)}")  # Open in browser

    
    elif selected_option == '8':
        while True:
            condition = input(
                translator.translate("Enter the condition (e.g., fever, cough, headache) or 'X' to go back: ",
                                     dest=lang).text).lower()

            if condition == 'x':  
                print(translator.translate("Going back to the main menu...", dest=lang).text)
                break

            if condition in medications:
                meds = medications[condition]
                response = translator.translate(f"For {condition}, you can take: {', '.join(meds)}.", dest=lang).text
            else:
                response = translator.translate("Sorry, I don't have information for that condition.", dest=lang).text

            print(f"\nBot: {response}\n")

    

    
    elif selected_option == '9':
        print("\nWelcome to the AI-Based Symptom Checker!")
        print("Please answer the following questions with 'Yes' or 'No'.")

        while True:
            try:
                
                fever = input("Do you have Fever? (Yes/No): ").strip().lower()
                cough = input("Do you have Cough? (Yes/No): ").strip().lower()
                fatigue = input("Do you have Fatigue? (Yes/No): ").strip().lower()
                breathing = input("Do you have Difficulty Breathing? (Yes/No): ").strip().lower()
                age = int(input("Enter your age (e.g., 25): ").strip())
                gender = input("Enter your gender (Male/Female): ").strip().lower()
                blood_pressure = input("Blood Pressure (Low/Normal/High): ").strip().lower()
                cholesterol = input("Cholesterol Level (Low/Normal/High): ").strip().lower()

                
                symptom_map = {"yes": 1, "no": 0}
                gender_map = {"male": 1, "female": 0}
                bp_map = {"low": 0, "normal": 1, "high": 2}
                cholesterol_map = {"low": 0, "normal": 1, "high": 2}

                user_input = {
                    'Fever': symptom_map.get(fever, 0),
                    'Cough': symptom_map.get(cough, 0),
                    'Fatigue': symptom_map.get(fatigue, 0),
                    'Difficulty Breathing': symptom_map.get(breathing, 0),
                    'Age': age,
                    'Gender': gender_map.get(gender, 0),
                    'Blood Pressure': bp_map.get(blood_pressure, 1),
                    'Cholesterol Level': cholesterol_map.get(cholesterol, 1)
                }

                
                user_input_df = pd.DataFrame([user_input])
                user_input_scaled = scaler.transform(user_input_df)

                
                prediction = model.predict(user_input_scaled)
                predicted_label = prediction[0]
                predicted_disease = label_encoder_y.inverse_transform([predicted_label])[0]

                
                prediction_proba = model.predict_proba(user_input_scaled)
                predicted_label = np.argmax(prediction_proba[0])  
                confidence = prediction_proba[0][predicted_label]  
                predicted_disease = label_encoder_y.inverse_transform([predicted_label])[0]
                
                if confidence < 0.5:
                    print(
                        "The model is uncertain about the diagnosis. Please consult a healthcare professional for further evaluation.")

                
                print(f"\nPredicted Illness: {predicted_disease}")
                print(f"Prediction Confidence: {confidence * 100:.2f}%")
                print(
                    "Note: This is an AI-based prediction. Please consult a healthcare professional for an accurate diagnosis.\n")



                
                exit_choice = input(
                    "Enter 'X' to return to the main menu, or press Enter to check again: ").strip().lower()
                if exit_choice == 'x':
                    break

            except Exception as e:
                print(f"Error: {e}. Please try again with valid inputs.\n")

    
    elif selected_option == '10':
        while True:
            condition = input(translator.translate(
                "Enter a condition (e.g., stress, anxiety, depression, grief, loneliness) or 'X' to go back: ",
                dest=lang).text).lower()

            if condition == 'x':  
                print(translator.translate("Going back to the main menu...", dest=lang).text)
                break

            
            tips = mental_health_tips[mental_health_tips['Condition'].str.lower() == condition]

            if not tips.empty:
                tip = tips.sample(1)['Tip'].values[0]  
                print(translator.translate(f"Here’s a tip for {condition}: {tip}", dest=lang).text)
            else:
                print(translator.translate("Sorry, I don't have tips for that condition. Please try another.",
                                           dest=lang).text)
    elif selected_option == '11':
        
        display_model_comparison()



    elif selected_option == '12':
        
        dynamic_model_selection()


    elif selected_option == '13':

        analyze_clusters()

    elif selected_option == '14':

        symptom_checker_naive_bayes()



    elif selected_option == "15":

        forecast_health_trends()

    if selected_option == "16":
        run_deep_learning_model()

    elif selected_option == '17':

        file_path = "Disease_symptom_and_patient_profile_dataset.csv"

        data = load_dataset(file_path)

        data = preprocess_data(data)

        X, _ = prepare_features_target(data)

        dbscan_clustering(X)

    elif selected_option == '18':

        file_path = "Disease_symptom_and_patient_profile_dataset.csv"

        data = load_dataset(file_path)

        data = preprocess_data(data)

        X, _ = prepare_features_target(data)

        hierarchical_clustering(X)

    elif selected_option == '19':

        file_path = "Disease_symptom_and_patient_profile_dataset.csv"

        data = load_dataset(file_path)

        data = preprocess_data(data)

        X, y = prepare_features_target(data)

        semi_supervised_learning(X, y)









    elif selected_option == '20':

        while True:

            user_input = input("\nEnter your symptoms (type 'X' to return to the main menu): ").strip()

            if user_input.lower() == 'x':
                print("Returning to the main menu...\n")

                break

            

            predicted_condition = nlp_user_symptoms()

            if predicted_condition:  

                print(f"\nBased on your symptoms, the predicted condition is: **{predicted_condition}**")

                

                more_info = input("Would you like to know more about this condition? (Yes/No): ").strip().lower()

                if more_info == 'yes':

                    interactive_conversation(predicted_condition)

                elif more_info == 'no':

                    print("Alright! Let me know if you need anything else.")

            else:

                print("Could not predict a condition. Please try again or train the model using option 22.")





    elif selected_option == '21':

        while True:

            print("\nRunning Sentiment Analysis...")

            user_input = input("Enter your thoughts or feedback (type 'X' to return to the main menu): ").strip()

            

            if user_input.lower() == 'x':
                print("Returning to the main menu...\n")

                break

            

            from textblob import TextBlob

            analysis = TextBlob(user_input)

            polarity = analysis.sentiment.polarity

            subjectivity = analysis.sentiment.subjectivity

            

            if polarity > 0:

                sentiment = "Positive"

            elif polarity < 0:

                sentiment = "Negative"

            else:

                sentiment = "Neutral"

            

            print(f"\nPolarity: {polarity:.2f} | Subjectivity: {subjectivity:.2f}")

            print(f"Sentiment: {sentiment}")


    elif selected_option == "22":

        train_nlp_model()


    
    elif selected_option == 'X':
      print(translator.translate("Application shutting down...", dest=lang).text)
      break

    else:
      print(translator.translate("\nPlease select a valid option\n", dest=lang).text)

# Test case: Any messages with "Cold" intent is not recognized -> fixed by adding the word "cold" in the intents.json
# Test case: Any messages with "Rash" intent is not recognized -> fixed by adding the words "rash" and "rashes" in the intents.json
# Test case: Messages for "Burn" intent returns "Sun burn" messages

