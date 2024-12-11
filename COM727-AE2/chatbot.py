
# package version control for PyCharm:
# Python == 3.10 (set in environment/interpreter)
# NumPy == latest (2.1.3 or 1.26.4)
# NLTK == 3.9.1
# TensorFlow == 2.14.1
# Keras == 2.14.0 (comes installed with TensorFlow)
# Googletrans == 3.1.0a0

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from googletrans import Translator

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

#Clean up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Converts the sentences into a bag of words
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
[X] Exit application

""", dest=lang).text, "\n")
    selected_option = input(translator.translate("You: ", dest=lang).text)
    return selected_option

print("QuinSpark Chatbot is here!\n")

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

    # Exit application
    elif selected_option == 'X':
      print(translator.translate("Application shutting down...", dest=lang).text)
      break

    else:
      print(translator.translate("\nPlease select a valid option\n", dest=lang).text)

# Test case: Any messages with "Cold" intent is not recognized -> fixed by adding the word "cold" in the intents.json
# Test case: Any messages with "Rash" intent is not recognized -> fixed by adding the words "rash" and "rashes" in the intents.json
# Test case: Messages for "Burn" intent returns "Sun burn" messages
