import abc
import random
import json
import pickle
import numpy as np
import os
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import nltk
from googletrans import Translator

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class QueryEngine(abc.ABC):
    """
    Abstract Base Class for Query Engine. This is used to plug into the HMI
    with variant Engines that conform to the same interface definition.
    """
    @abc.abstractmethod
    def prompt(self, message:str) -> str:
        """
        Prompt for User Input
        :param message: Message to be sent to the Chat-Bot
        :return: Response to Message (Blocking API)
        """
        pass

    @abc.abstractmethod
    def new_session(self) -> str:
        """
        Initialises the Session
        :return: Statement from Chat-Bot to begin the session
        """
        pass

    @abc.abstractmethod
    def reset_session(self) -> str:
        """
        Resets the Session
        :return: Statement from Chat-Bot to reset the session
        """
        pass


class FirstAidEngineException(Exception):
    pass


class FirstAidEngine(QueryEngine):

    ERROR_THRESHOLD = 0.25

    def __init__(self, data_path="./data/", model_path="./model/", lang='en', debug=False):
        """
        :param data_path:   Prefix Path for Data Sets
        :param model_path:  Prefix Path for Models
        :param lang:        Default Language
        :param debug:       Verbose Mode
        """

        self._data_path = data_path
        self._model_path = model_path
        self._lang = lang
        self._debug = debug
        self._initialised = False

        # NTLK Lemmatizer
        self._lemmatizer = WordNetLemmatizer()

        # Data Set
        self._intents = json.loads(open('./data/intents.json').read())

        # Model
        self._instructions_words = pickle.load(open('./model/instructions_words.pkl', 'rb'))
        self._instructions_classes = pickle.load(open('./model/instructions_classes.pkl', 'rb'))
        self._instructions_model = load_model('./model/ANN_instructions_model.keras')

        # Language Translation
        self._translator = Translator()

        self._language_list = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy',
                               'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn',
                               'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny',
                               'chinese (simplified)': 'zh-cn', 'chinese (traditional)': 'zh-tw',
                               'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dutch': 'nl',
                               'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl',
                               'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka',
                               'german': 'de', 'greek': 'el', 'gujarati': 'gu', 'haitian creole': 'ht',
                               'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'he', 'hindi': 'hi', 'hmong': 'hmn',
                               'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'indonesian': 'id',
                               'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn',
                               'kazakh': 'kk', 'khmer': 'km', 'korean': 'ko', 'kurdish': 'ku', 'kurmanji': 'ku',
                               'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lithuanian': 'lt',
                               'luxembourgish': 'lb', 'macedonian': 'mk', 'malagasy': 'mg', 'malay': 'ms',
                               'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'mongolian': 'mn',
                               'myanmar': 'my', 'burmese': 'my', 'nepali': 'ne', 'norwegian': 'no',
                               'odia': 'or', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt',
                               'punjabi': 'pa', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm',
                               'scots gaelic': 'gd', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd',
                               'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so',
                               'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg',
                               'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr',
                               'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi',
                               'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}

        self._initialised = True

    def __clean_up_sentence(self, sentence: str):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self._lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words


    def __bag_of_words(self, sentence: str) -> np.array:
        sentence_words = self.__clean_up_sentence(sentence)
        bag = [0] * len(self._instructions_words)
        for w in sentence_words:
            for i, word in enumerate(self._instructions_words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def __predict_class(self, sentence: str) -> list[dict]:
        """
        Prediction Class
        :param sentence: The sentence that needs to be predicted
        :return: The Response
        """
        bow = self.__bag_of_words(sentence)  # bow: Bag Of Words, feed the data into the neural network
        res = self._instructions_model.predict(np.array([bow]), verbose=0)[0]  # res: result. [0] as index 0
        # Senju - Notes: added verbose=0 to hide the predict progress bar

        results = [[i, r] for i, r in enumerate(res) if r > self.ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self._instructions_classes[r[0]], 'probability': str(r[1])})
        return return_list

    @staticmethod
    def __get_response(intents_list: list, intents_json: dict):
        """
        Internal function to return the response from the Model
        :param intents_list: List of Intents
        :param intents_json: JSON file containing the actual response text to use
        :return: Result List
        """
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result, tag

    def __translate(self, prompt: str, lang: str = "") -> str:
        """
        Translation Wrapper
        :param prompt: The message to be translated
        :param lang: Language override. If empty the default language will be used
        :return: The translated message
        """
        translation_language = lang if len(lang) > 0 else self._lang
        return self._translator.translate(prompt, translation_language).text

    def prompt(self, message:str) -> str:
        """
        Prompts the Patient for data entry
        :param message: The message from the patient
        :return: The result of the chatbot. Note this is a **blocking** API
        """

        if not self._initialised:
            raise FirstAidEngineException("First Aid Engine Not Initialised")

        translated_message = self.__translate(message, "en")
        intents = self.__predict_class(translated_message)

        if intents is not None:
            res, tag = self.__get_response(intents, self._intents)
            translated_tag = self.__translate("Instruction for " + tag)
            translated_result = self.__translate(res)
            return translated_tag + ":\n " + translated_result
        else:
            return self.__translate("I'm sorry, I didn't get that.")


    def new_session(self) -> str:
        """
        Starts a New Session with the Chat-Bot
        :return:
        """
        if not self._initialised:
            raise FirstAidEngineException("First Aid Engine Not Initialised")

        return self.__translate("I can provide you with first aid instructions, how can I help you?")


    def reset_session(self) -> str:
        """Wrapper around the New Session"""
        return self.new_session()