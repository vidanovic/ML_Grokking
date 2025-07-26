# house_predictions.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

def main():
    data = pd.read_csv('../data/Hyderabad.csv')
    X = data.drop('Price', axis=1)
    y = data['Price']

    # One-hot encode categorical columns automatically
    X_encoded = pd.get_dummies(X)

    model = LinearRegression()
    model.fit(X_encoded, y)

    # Calculate and print RMSE on training data
    from sklearn.metrics import mean_squared_error
    y_pred = model.predict(X_encoded)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"Root Mean Square Error (RMSE): {rmse}")

    # Example new house data
    new_house = pd.DataFrame({'Area': [1000], 'No. of Bedrooms': [3]})

    # One-hot encode using the same columns as training data
    new_house_encoded = pd.get_dummies(new_house)
    new_house_encoded = new_house_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Predict price
    predicted_price = model.predict(new_house_encoded)
    print(f'predicted_price: {predicted_price}')

if __name__ == '__main__':
    main()
