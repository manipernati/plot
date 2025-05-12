import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Data
x = np.array([151, 174, 138, 186, 128, 136, 179, 163, 152, 131]).reshape(-1, 1)
y = np.array([63, 81, 56, 91, 47, 57, 76, 72, 62, 48])

# Create and train model
model = LinearRegression()
model.fit(x, y)

# Coefficients
slope = model.coef_[0]
intercept = model.intercept_
print(f"Linear Regression Equation: y = {slope:.2f}x + {intercept:.2f}")

# Predictions
y_pred = model.predict(x)

# Accuracy metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Predict for a new value
new_x = np.array([[160]])
predicted_y = model.predict(new_x)
print(f"Predicted y for x = {new_x[0][0]}: {predicted_y[0]:.2f}")

# Plotting
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.scatter(new_x, predicted_y, color='green', marker='x', s=100, label='Predicted Point')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
