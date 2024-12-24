# COM727-AE2

## How to access the chatbot application on PyCharm:

1. Create a project in PyCharm
2. Create an appropriate interpreter / environment (refer to "package_versions.txt")
3. Copy and paste "ANN_training.py", "chatbot.py", "intents.json" into your project
4. Run "ANN_training.py"
5. Run "chatbot.py"

(Any error occured while running could most likely be due to version control)

## To install required libraries
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn joblib nltk textblob imblearn statsmodels customtkinter folium selenium osmnx

## How to access the chatbot application on Google Colab:

1. Create a new notebook in Google Colab
2. Upload "intents.json" and "health_tips.json" under "content" folder in Google Colab
3. Copy and paste codes from "ANN_training.py", "chatbot.py" into your notebook (seperate cells!)
4. Comment out the "Use for PyCharm" codes and un-comment the "Use for Google Colab" codes
5. Run "ANN_training.py"
6. Run "chatbot.py"


