import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def load_time_series_data():
    print("Loading Time Series Data...")
    data = pd.read_csv("time_series_symptoms.csv", parse_dates=["Date"], index_col="Date")
    data.index = pd.date_range(start=data.index[0], periods=len(data), freq='D')
    return data


def visualize_historical_data(data):
    print("Visualizing Historical Symptom Data...")
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Symptom_Count'], marker='o', label="Observed Symptom Count")
    plt.title("Health Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Symptom Count")
    plt.legend()
    plt.show()


def forecast_health_trends(data):
    print("Forecasting Future Health Trends...")
    model = ExponentialSmoothing(data['Symptom_Count'], trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)

    
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Symptom_Count'], marker='o', label="Observed Symptom Count")
    plt.plot(pd.date_range(start=data.index[-1], periods=7, freq='ME')[1:], forecast, color="red", marker='o',
             label="Forecasted Trends")
    plt.title("Health Trends Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Symptom Count")
    plt.legend()
    plt.show()
    return forecast


def forecast_health_trends_interactive():
    data = load_time_series_data()
    visualize_historical_data(data)
    forecast = forecast_health_trends(data)

    print("\nWould you like insights on the trends?")
    print("\nHere are some questions you can ask:")
    print("[1] What does this data mean?")
    print("[2] What can I infer from the forecast?")
    print("[3] What is the next peak prediction?")
    print("[4] How accurate is this forecast?")
    print("[X] Exit to Main Menu")

    while True:
        choice = input("Enter the number of your question or 'X' to return: ").strip().lower()

        if choice == '1':
            print("\nWhat does this data mean?")
            print("Answer: The graph shows historical trends of symptoms reported over time. Peaks indicate periods of increased illness.\n")
        elif choice == '2':
            print("\nWhat can I infer from the forecast?")
            print("Answer: The forecast predicts future increases or decreases in symptom counts based on historical patterns.\n")
        elif choice == '3':
            print("\nWhat is the next peak prediction?")
            print("Answer: Based on the forecast, symptom counts are expected to rise within the next few months.\n")
        elif choice == '4':
            print("\nHow accurate is this forecast?")
            print("Answer: The forecast uses Holt-Winters Exponential Smoothing, suitable for seasonal trends. Accuracy depends on data quality.\n")
        elif choice == 'x':
            print("Returning to the Main Menu...\n")
            break
        else:
            print("Invalid choice. Please select a valid option.")



def main():
    forecast_insights()


if __name__ == "__main__":
    main()
