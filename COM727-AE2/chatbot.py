# package version control for PyCharm:
# Python == 3.10 (set in environment/interpreter)
# NumPy == latest (2.1.3 or 1.26.4)
# NLTK == 3.9.1
# TensorFlow == 2.14.1
# Keras == 2.14.0 (comes installed with TensorFlow)

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
health_tips = json.loads(open('health_tips.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('ANN_chatbot_model.keras')


# Clean up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Converts the sentences into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)  # bow: Bag Of Words, feed the data into the neural network
    res = model.predict(np.array([bow]), verbose=0)[0]  # res: result. [0] as index 0
    # Senju - Notes: added verbose=0 to hide the predict progress bar
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
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
    selected_option = input("""Please select one of the following options:
[1] Symptom checker - N/A
[2] First aid instructions
[3] Display emergency contacts
[4] Display health tips
[5] Display resource links - N/A
[X] Exit application

""")
    return selected_option


print("COM727 Chatbot is here!\n")

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
        print("\nI can provide you with first aid instructions, "
              "how can I help you? (Enter \"X\" to go back to menu)\n")
        while True:
            message = input("You: ")
            if message == 'X':
                print("\nGoing back to menu...\n")
                break
            else:
                ints = predict_class(message)
                if ints:  # Senju - Notes: added if-else for error-handling
                    res, tag = get_response(ints, intents)
                    res = "\033[1mInstruction for " + tag + "\033[0m\n" + res + "\n\nCan I help you with anything else?"
                else:
                    res = "Sorry, I didn't get that. \nCan I help you with anything else?"
                print(f"\n{res}\n")

    # Display emergency contacts
    elif selected_option == '3':
        while True:
            print("""\n
\033[1mEmergency & First Aid Hotline Numbers:\033[0m

In case of an emergency or if you need immediate medical assistance, here are the important numbers you should know:

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

""")

            message = input("You: ")
            if message == 'X':
                print("\nGoing back to menu...\n")
                break
            else:
                continue

    # Display health tips
    elif selected_option == '4':
        while True:
            rnd = str(random.randint(1, 10))
            for i in health_tips['intents']:
                if rnd in i['patterns']:
                    res = "\033[1m\033[3m" + i['responses'][0] + "\033[0m" + i['responses'][
                        1] + "\n\n" + "\033[1m\033[3m" + i['responses'][2] + "\033[0m" + i['responses'][3]
                    tag = i['tag'][0]
                    break
            res = "\033[1m\033[4m" + tag + "\n\033[0m\n" + res + "\n\n\n\nShould I generate another response? (Y/N)"
            print(f"\n{res}\n")

            message = input("You: ")
            if message == 'Y':
                print("\n\n")
                continue
            else:
                print("\nGoing back to menu...\n")
                break

    # Display resource links
    elif selected_option == '5':
        while True:
            print("""\n
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

""")

            message = input("You: ")
            if message == 'X':
                print("\nGoing back to menu...\n")
                break
            else:
                continue

    # Exit application
    elif selected_option == 'X':
        print("Application shutting down...")
        break

    else:
        print("\nPlease select a valid option\n")

# Test case: Any messages with "Cold" intent is not recognized -> fixed by adding the word "cold" in the intents.json
# Test case: Any messages with "Rash" intent is not recognized -> fixed by adding the words "rash" and "rashes" in the intents.json
# Test case: Messages for "Burn" intent returns "Sun burn" messages
