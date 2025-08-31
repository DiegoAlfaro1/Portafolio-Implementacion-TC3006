import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducible results
np.random.seed(42)

# ========================================================================================
# DATA LOADING AND PREPROCESSING
# ========================================================================================

# Load the coffee sales dataset
dataset = "Coffe_sales.csv"
df = pd.read_csv(dataset)

# Data cleaning and feature engineering:
# 1. Remove redundant columns (Month_name, Time, Weekday, cash_type, coffee_name)
# 2. Rename columns for clarity
# 3. Keep only relevant features for time-based price prediction

# Remove unnecessary columns
df = df.drop(columns=['Month_name', 'Time', 'Weekday', 'cash_type', 'coffee_name'])

# Rename columns for better readability
df.rename(columns={
    'money': 'Price', 
    'Time_of_Day': 'Time_of_day', 
    'hour_of_day': 'Hour_of_day'
}, inplace=True)

# ========================================================================================
# TRAIN-TEST SPLIT
# ========================================================================================

# Split dataset into training (70%) and testing (30%) sets
df_train = df.sample(frac=0.7, random_state=42)
df_test = df.drop(df_train.index)

# Extract features: Hour of day, Day of week, Month
# These temporal features will be used to predict coffee prices
x_train = df_train[['Hour_of_day', "Weekdaysort", "Monthsort"]].values.astype(float)
x_test = df_test[['Hour_of_day', "Weekdaysort", "Monthsort"]].values.astype(float)

# Extract target variable: Coffee price
y_train = df_train['Price'].values.astype(float)
y_test = df_test['Price'].values.astype(float)

# ========================================================================================
# FEATURE STANDARDIZATION
# ========================================================================================

# Standardize features to have mean=0 and std=1
# This helps gradient descent converge faster and more reliably
X_mean = x_train.mean(axis=0)  # Calculate mean from training data only
X_std = x_train.std(axis=0)    # Calculate std from training data only

# Apply standardization using training statistics to both sets
x_train_scaled = (x_train - X_mean) / X_std
x_test_scaled = (x_test - X_mean) / X_std   # Use training stats to avoid data leakage

# Add bias column (column of 1s) for the intercept term in linear regression
x_train_with_bias = np.column_stack([np.ones(x_train_scaled.shape[0]), x_train_scaled])
x_test_with_bias = np.column_stack([np.ones(x_test_scaled.shape[0]), x_test_scaled])

# ========================================================================================
# MODEL IMPLEMENTATION - MULTIPLE LINEAR REGRESSION FROM SCRATCH
# ========================================================================================

# Initialize weights (theta) with small random values
# Shape: [bias_weight, hour_weight, weekday_weight, month_weight]
weight = np.random.normal(0, 0.01, 4)

def predict(X, weight):
    """
    Make predictions using linear regression
    Formula: y = X @ theta (matrix multiplication)
    
    Args:
        X: Feature matrix with bias column (n_samples, 4)
        weight: Weight vector (4,)
    
    Returns:
        predictions: Predicted values (n_samples,)
    """
    return X @ weight

def cost(X, y, weight):
    """
    Calculate Mean Squared Error cost function
    Formula: J = (1/2m) * sum((predicted - actual)^2)
    
    Args:
        X: Feature matrix (n_samples, 4)
        y: True values (n_samples,)
        weight: Weight vector (4,)
    
    Returns:
        cost: Single cost value (lower is better)
    """
    predictions = predict(X, weight)
    difference = (predictions - y) ** 2
    return (1/2) * np.mean(difference)

def gradient(X, y, weight):
    """
    Calculate gradients for weight updates
    Formula: grad = (1/m) * X.T @ (predicted - actual)
    
    Args:
        X: Feature matrix (n_samples, 4)
        y: True values (n_samples,)
        weight: Current weights (4,)
    
    Returns:
        gradients: Gradient vector (4,) - tells us how to update each weight
    """
    predictions = predict(X, weight)
    error = predictions - y
    return (1/len(y)) * X.T @ error

# ========================================================================================
# GRADIENT DESCENT TRAINING
# ========================================================================================

# Training hyperparameters
learning_rate = 0.01    # Step size for weight updates
max_epochs = 1000       # Maximum number of training iterations
tolerance = 1e-6        # Stop if cost improvement is smaller than this

# Initialize training tracking
cost_history = []       # Track cost over time to monitor convergence
theta = weight.copy()   # Current weights (will be updated during training)

print("=== Training Multiple Linear Regression ===")

# Training loop - Gradient Descent Algorithm
for epoch in range(max_epochs):
    # Calculate current cost
    current_cost = cost(x_train_with_bias, y_train, theta)
    cost_history.append(current_cost)
    
    # Calculate gradients (how much to change each weight)
    grads = gradient(x_train_with_bias, y_train, theta)
    
    # Update weights using gradient descent rule: theta = theta - alpha * gradient
    theta = theta - learning_rate * grads
    
    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cost = {current_cost:.4f}")
    
    # Early stopping: if cost stops improving significantly, stop training
    if len(cost_history) > 1:
        if abs(cost_history[-2] - cost_history[-1]) < tolerance:
            print(f"Converged at epoch {epoch}")
            break

print(f"Final training cost: {cost_history[-1]:.4f}")
print(f"Learned weights: {theta}")
print(f"Model equation: Price = {theta[0]:.2f} + {theta[1]:.2f}*hour + {theta[2]:.2f}*weekday + {theta[3]:.2f}*month")

# ========================================================================================
# MODEL EVALUATION
# ========================================================================================

# Make predictions on test set
test_predictions = predict(x_test_with_bias, theta)
test_cost = cost(x_test_with_bias, y_test, theta)

print(f"\nTest set cost: {test_cost:.4f}")

def accuracy_by_price_range(predictions, actual):
    """Analyze model accuracy across different price ranges"""
    low_range = actual <= 25     # Budget coffees
    mid_range = (actual > 25) & (actual <= 35)  # Standard coffees  
    high_range = actual > 35     # Premium coffees
    
    print("=== Accuracy by Price Range ===")
    for range_name, mask in [("Low ($18-25)", low_range), 
                            ("Mid ($25-35)", mid_range), 
                            ("High ($35+)", high_range)]:
        if np.sum(mask) > 0:
            range_mae = np.mean(np.abs(predictions[mask] - actual[mask]))
            range_rmse = np.sqrt(np.mean((predictions[mask] - actual[mask])**2))
            count = np.sum(mask)
            print(f"{range_name}: {count} samples, MAE: ${range_mae:.2f}, RMSE: ${range_rmse:.2f}")

def accuracy_within_threshold(predictions, actual, thresholds=[1, 2, 3, 5]):
    """Calculate percentage of predictions within acceptable error ranges"""
    print("\n=== Predictions Within Error Thresholds ===")
    errors = np.abs(predictions - actual)
    
    for threshold in thresholds:
        within_threshold = np.sum(errors <= threshold)
        percentage = (within_threshold / len(errors)) * 100
        print(f"Within ${threshold}: {within_threshold}/{len(errors)} ({percentage:.1f}%)")

def compare_to_baselines(predictions, actual):
    """Compare model performance to simple baseline models"""
    print("\n=== Comparison to Baseline Models ===")
    
    # Baseline 1: Always predict the mean price
    mean_baseline = np.full(len(actual), np.mean(actual))
    mean_mae = np.mean(np.abs(mean_baseline - actual))
    mean_rmse = np.sqrt(np.mean((mean_baseline - actual)**2))
    
    # Baseline 2: Always predict the median price
    median_baseline = np.full(len(actual), np.median(actual))
    median_mae = np.mean(np.abs(median_baseline - actual))
    median_rmse = np.sqrt(np.mean((median_baseline - actual)**2))
    
    # Our trained model
    model_mae = np.mean(np.abs(predictions - actual))
    model_rmse = np.sqrt(np.mean((predictions - actual)**2))
    
    print(f"Your Model    - MAE: ${model_mae:.2f}, RMSE: ${model_rmse:.2f}")
    print(f"Mean Baseline - MAE: ${mean_mae:.2f}, RMSE: ${mean_rmse:.2f}")
    print(f"Median Baseline - MAE: ${median_mae:.2f}, RMSE: ${median_rmse:.2f}")
    
    # Calculate improvement percentages
    improvement_vs_mean = ((mean_mae - model_mae) / mean_mae) * 100
    improvement_vs_median = ((median_mae - model_mae) / median_mae) * 100
    
    print(f"\nImprovement over mean baseline: {improvement_vs_mean:.1f}%")
    print(f"Improvement over median baseline: {improvement_vs_median:.1f}%")

# Run all evaluation metrics
accuracy_by_price_range(test_predictions, y_test)
accuracy_within_threshold(test_predictions, y_test)
compare_to_baselines(test_predictions, y_test)

# ========================================================================================
# VISUALIZATION
# ========================================================================================

# Create comprehensive visualization of model performance
plt.figure(figsize=(15, 10))

# Plot 1: Predicted vs Actual prices
plt.subplot(2, 3, 1)
plt.scatter(y_test, test_predictions, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Predicted vs Actual Prices')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals (prediction errors)
plt.subplot(2, 3, 2)
residuals = test_predictions - y_test
plt.scatter(test_predictions, residuals, alpha=0.6, color='red')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.8)
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residual (Predicted - Actual)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Plot 3: Distribution of prediction errors
plt.subplot(2, 3, 3)
plt.hist(np.abs(residuals), bins=25, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Absolute Error ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.grid(True, alpha=0.3)

# Plot 4: Training progress (cost vs iterations)
plt.subplot(2, 3, 4)
plt.plot(cost_history, color='purple', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Training Progress')
plt.grid(True, alpha=0.3)

# Plot 5: Feature importance (weight magnitudes)
plt.subplot(2, 3, 5)
feature_names = ['Bias', 'Hour', 'Weekday', 'Month']
colors = ['red' if w < 0 else 'blue' for w in theta]
plt.bar(feature_names, np.abs(theta), color=colors, alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Weight Magnitude')
plt.title('Feature Importance (Weight Magnitudes)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 6: Actual vs predicted price distributions
plt.subplot(2, 3, 6)
plt.hist(y_test, bins=25, alpha=0.5, label='Actual Prices', color='blue', density=True)
plt.hist(test_predictions, bins=25, alpha=0.5, label='Predicted Prices', color='red', density=True)
plt.xlabel('Price ($)')
plt.ylabel('Density')
plt.title('Price Distribution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================================================================
# SUMMARY STATISTICS
# ========================================================================================

print("\n" + "="*70)
print("FINAL MODEL SUMMARY")
print("="*70)
print(f"Training samples: {len(x_train)} | Test samples: {len(x_test)}")
print(f"Features used: Hour of day, Day of week, Month")
print(f"Target variable: Coffee price (${y_test.min():.2f} - ${y_test.max():.2f})")
print(f"Final training cost: {cost_history[-1]:.4f}")
print(f"Test set cost: {test_cost:.4f}")
print(f"Model converged in {len(cost_history)} iterations")

# Calculate R-squared score
y_mean = np.mean(y_test)
ss_total = np.sum((y_test - y_mean) ** 2)
ss_residual = np.sum((test_predictions - y_test) ** 2)
r2_score = 1 - (ss_residual / ss_total)
print(f"R-squared score: {r2_score:.4f} ({r2_score*100:.1f}% of variance explained)")

print("\nModel Interpretation:")
print(f"- Base price: ${theta[0]:.2f}")
print(f"- Hour effect: ${theta[1]:.2f} per hour")
print(f"- Weekday effect: ${theta[2]:.2f} per day")
print(f"- Month effect: ${theta[3]:.2f} per month")
print("="*70)