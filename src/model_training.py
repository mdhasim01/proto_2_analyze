import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to load preprocessed data

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to train and evaluate a linear regression model

def train_evaluate_model(data):
    # Split data into features and target
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    
    return model

# Example usage
if __name__ == '__main__':
    file_path = '/home/hasim001/proto_analyse/data/preprocessed_data.csv'
    data = load_data(file_path)
    model = train_evaluate_model(data)
