
# Import package
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import cross_val_score


# Record start time
start_time = time.time()

# Step 1: Load and preprocess the data
train_data = pd.read_excel('E:\\GitHubData\\input1_sc-xgb_TrainFit_data.xlsx')
validation_data = pd.read_excel('E:\\GitHubData\\input2_sc-xgb_ValidationTest_data.xlsx')

# Data partitioning
X_train = train_data.iloc[:, :-8]
y_train = train_data.iloc[:, -8:]
X_val = validation_data.iloc[:, :-8]
y_val = validation_data.iloc[:, -8:]

# View statistical information of the target variable
print("Training Target Statistics:")
print(y_train.describe())
print("\nValidation Target Statistics:")
print(y_val.describe())

# Standardized data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val)

# Convert standardized data into XGBoost specific format DMatrix to improve model performance
dtrain = xgb.DMatrix(X_train_scaled, label=y_train_scaled)
dval = xgb.DMatrix(X_val_scaled, label=y_val_scaled)

# Extract input features from training data to calculate the water balance in the loss function
QQ_train = train_data[['QL(R=1)', 'QL(R=2)', 'QL(R=3)', 'QU(R=1)', 'QU(R=2)', 'QU(R=3)','V(R=1)', 'V(R=2)', 'V(R=3)',\
                       'Vend1', 'Vend2', 'Vend3', 'V_hu', 'WI1', 'WI2', 'WI3', 'WI4', 'QL_hu', 'V_huend', 'QO(R=1)',\
                       'QO(R=2)', 'QO(R=3)', 'QO_hu', 'WO1', 'WO2', 'WO3', 'WO4']]


max_values = QQ_train.max()
min_values = QQ_train.min()

# Step 2: Customize the loss function and perform matrix operations directly on y_pred and y_true,
# #The parameters of the loss function only include the predicted value y_pred of the model and the data object dtrain, and cannot directly access external variables.
# #QQ_train, max-values, and min-values need to be set as global variables so that these variables can be accessed within the custom-loss_maultivariant function,
# #Thus, calculate the water balance error and inverse normalization.
# #The role of custom loss function in XGBoost is to guide the model optimization objectives.
# #It tells the model how to adjust parameters (leaf node splitting) to minimize losses by calculating gradients and Hessian (second derivative).
# #In the training phase: During each iteration, XGBoost uses a custom loss function to calculate the error between the current model prediction value
# and the target value, and updates the tree structure based on gradients and Hessian to gradually approach the target value.
# #In the validation phase, the loss value of the validation set is only used to evaluate the performance of the model on unseen data and does not participate in gradient updates.
# #If you only specify eval_metric (such as' RMSE ') in params, the loss value of the validation set is calculated based on RMSE or other default metrics.
# #The predicted values on the validation set are fixed and calculated by the model.
# #The calculation of the loss value can be based on eval_tetric or other metrics such as RMSE, MAE, R ², and is not directly related to the custom loss function used during training.

def custom_loss_multivariate(y_pred, dtrain):
    # Obtain the true value matrix
    y_true = dtrain.get_label().reshape(y_pred.shape)

    # The calculation of the loss function is based on the entire training set (dtrain).
    # Inverse normalization of y_pred to its original value
    def inverse_transform(y_pred_column, col_name):
        return y_pred_column * (max_values[col_name] - min_values[col_name]) + min_values[col_name]

    QO_R1 = inverse_transform(y_pred[:, 0], 'QO(R=1)') # Normalize the normalized predicted values according to the function above
    QO_R2 = inverse_transform(y_pred[:, 1], 'QO(R=2)')
    QO_R3 = inverse_transform(y_pred[:, 2], 'QO(R=3)')
    QO_R4 = inverse_transform(y_pred[:, 3], 'QO_hu')
    WO1 = inverse_transform(y_pred[:, 4], 'WO1')
    WO2 = inverse_transform(y_pred[:, 5], 'WO2')
    WO3 = inverse_transform(y_pred[:, 6], 'WO3')
    WO4 = inverse_transform(y_pred[:, 7], 'WO4')

    # Initialize gradient and Hessian
    residual = y_true - y_pred  # Residual
    gradient = -2 * residual  # Basic mean square error gradient
    hessian = 2 * np.ones_like(residual)  # Basic mean square error Hessian

    # Negative penalty term, this must be present. To reduce negative values, all four models must have it
    penalty = beta * np.sum(np.square(np.minimum(0, y_pred)), axis=1)
    grad_penalty = beta * 2 * np.minimum(0, y_pred)  # Gradient of negative values
    hess_penalty = beta * 2 * (y_pred < 0).astype(float)  # Hessian in the negative value


    # Add negative penalty term to gradient and Hessian
    gradient += grad_penalty
    hessian += hess_penalty

    #Water balance formula 1
    e_R1 = (QQ_train['QU(R=1)'].values - QO_R1) * Coef + (QQ_train['WI1'].values - WO1)- (QQ_train['V(R=1)'].values - QQ_train['Vend1'].values)
    grad_R1_QO1 = -2 * e_R1 * Coef * (max_values['QO(R=1)'] - min_values['QO(R=1)'])
    grad_R1_WO1 = -2 * e_R1 * (max_values['WO1'] - min_values['WO1'])
    hess_R1_QO1 = 2 * (Coef * (max_values['QO(R=1)'] - min_values['QO(R=1)'])) ** 2
    hess_R1_WO1 = 2 * ((max_values['WO1'] - min_values['WO1'])) ** 2

    gradient[:, 0] += alpha1 * grad_R1_QO1
    gradient[:, 4] += alpha5 * grad_R1_WO1
    hessian[:, 0] += alpha1 * hess_R1_QO1
    hessian[:, 4] += alpha5 * hess_R1_WO1

    # Water balance formula 2
    e_R2 = (QQ_train['QU(R=2)'].values - QO_R2) * Coef + (QQ_train['WI2'].values - WO2) - (QQ_train['V(R=2)'].values - QQ_train['Vend2'].values)
    grad_R2_QO2 = -2 * e_R2 * Coef * (max_values['QO(R=2)'] - min_values['QO(R=2)'])
    grad_R2_WO2 = -2 * e_R2 * (max_values['WO2'] - min_values['WO2'])
    hess_R2_QO2 = 2 * (Coef * (max_values['QO(R=2)'] - min_values['QO(R=2)'])) ** 2
    hess_R2_WO2 = 2 * ((max_values['WO2'] - min_values['WO2'])) ** 2

    gradient[:, 1] += alpha2 * grad_R2_QO2
    gradient[:, 5] += alpha6 * grad_R2_WO2
    hessian[:, 1] += alpha2 * hess_R2_QO2
    hessian[:, 5] += alpha6 * hess_R2_WO2

    # Water balance formula 3
    e_R3 = (QQ_train['QU(R=3)'].values - QO_R3) * Coef + (QQ_train['WI3'].values - WO3) - (QQ_train['V(R=3)'].values - QQ_train['Vend3'].values)
    grad_R3_QO3 = -2 * e_R3 * Coef * (max_values['QO(R=3)'] - min_values['QO(R=3)'])
    grad_R3_WO3 = -2 * e_R3 * (max_values['WO3'] - min_values['WO3'])
    hess_R3_QO3 = 2 * (Coef * (max_values['QO(R=3)'] - min_values['QO(R=3)'])) ** 2
    hess_R3_WO3 = 2 * ((max_values['WO3'] - min_values['WO3'])) ** 2

    gradient[:, 2] += alpha3 * grad_R3_QO3
    gradient[:, 6] += alpha7 * grad_R3_WO3
    hessian[:, 2] += alpha3 * hess_R3_QO3
    hessian[:, 6] += alpha7 * hess_R3_WO3

    # Water balance formula 4
    e_R4 = (QO_R1 + QO_R2 + QO_R3 + QQ_train['QL(R=1)'].values + QQ_train['QL(R=2)'].values + QQ_train['QL(R=3)'].values+ QQ_train['QL_hu'].values\
            - QO_R4) * Coef + (QQ_train['WI4'].values - WO4) - ( QQ_train['V_hu'].values - QQ_train['V_huend'].values)
    grad_R4_QO1 = 2 * e_R4 * Coef * (max_values['QO(R=1)'] - min_values['QO(R=1)'])
    grad_R4_QO2 = 2 * e_R4 * Coef * (max_values['QO(R=2)'] - min_values['QO(R=2)'])
    grad_R4_QO3 = 2 * e_R4 * Coef * (max_values['QO(R=3)'] - min_values['QO(R=3)'])
    grad_R4_QO4 = -2 * e_R4 * Coef * (max_values['QO_hu'] - min_values['QO_hu'])
    grad_R4_WO4 = -2 * e_R4 * (max_values['WO4'] - min_values['WO4'])

    hess_R4_QO1 = 2 * (Coef * (max_values['QO(R=1)'] - min_values['QO(R=1)'])) ** 2
    hess_R4_QO2 = 2 * (Coef * (max_values['QO(R=2)'] - min_values['QO(R=2)'])) ** 2
    hess_R4_QO3 = 2 * (Coef * (max_values['QO(R=3)'] - min_values['QO(R=3)'])) ** 2
    hess_R4_QO4 = 2 * (Coef * (max_values['QO_hu'] - min_values['QO_hu'])) ** 2
    hess_R4_WO4 = 2 * ((max_values['WO4'] - min_values['WO4'])) ** 2

    gradient[:, 0] += alpha1 * grad_R4_QO1
    gradient[:, 1] += alpha2 * grad_R4_QO2
    gradient[:, 2] += alpha3 * grad_R4_QO3
    gradient[:, 3] += alpha4 * grad_R4_QO4
    gradient[:, 7] += alpha8 * grad_R4_WO4

    hessian[:, 0] += alpha1 * hess_R4_QO1
    hessian[:, 1] += alpha2 * hess_R4_QO2
    hessian[:, 2] += alpha3 * hess_R4_QO3
    hessian[:, 3] += alpha4 * hess_R4_QO4
    hessian[:, 7] += alpha8 * hess_R4_WO4

    # Return gradient and second derivative
    return gradient.ravel(), hessian.ravel()


# Step 3: Train and evaluate the model
train_predictions = []
val_predictions = []


# Parameters obtained from Bayesian optimization coupled subsample cross validation method
params = {
     'learning_rate': 0.04,
     'max_depth': 6,
     'subsample': 0.61,         # 80% of the data is used in each round for building the tree, and this sampling mechanism
                                # only affects the generation of the tree without changing the data in the loss function.
     'colsample_bytree': 0.94,  # Use 80% of the features in each round
     'seed': 80,                # Number of decision trees
     'eval_metric': 'rmse',     # Set evaluation indicators
     'min_child_weight': 7,     # Improve the regularization strength of the model
     'reg_lambda': 1
}


# Initialize evals_ result dictionary
evals_result = {}

# Training model
evals = [(dtrain, 'train'), (dval, 'validation')]
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=5000,
    evals=evals,
    early_stopping_rounds=100,
    verbose_eval=100, # Print log information every 100 iterations, including evaluation metrics for the training and validation sets (such as RMSE).
    evals_result=evals_result,  # Used to save the evaluation results of each round
    obj=custom_loss_multivariate  # Passing custom loss function
)

# Obtain the training and validation loss values for each round
train_losses = evals_result['train']['rmse']  # The RMSE of each round in the training set is an evaluation metric
val_losses = evals_result['validation']['rmse']  # Verify the RMSE for each round of the validation set

# Model training set prediction, using dtrain, aims to obtain the prediction results of the training set,
# and then compare them with the real values to calculate the training error.
train_predictions = model.predict(dtrain).reshape(y_train_scaled.shape)

# Model validation set prediction, using dval, aims to obtain the prediction results of the validation set and evaluate the generalization ability of the model.
# During the validation phase, the model automatically calculates the validation loss using the prediction results of dval,
# but the validation loss does not participate in the gradient update.
val_predictions = model.predict(dval).reshape(y_val_scaled.shape)

scores = cross_val_score(xgb.XGBRegressor(**params), X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation Scores:", scores)

# Anti standardization
train_predictions_rescaled = scaler_y.inverse_transform(train_predictions)
val_predictions_rescaled = scaler_y.inverse_transform(val_predictions)


# Step 5: Evaluate the model
train_metrics = {
    'RMSE': [],
    'R²': [],
    'MAE': [],
    'Explained Variance': []
}
val_metrics = {
    'RMSE': [],
    'R²': [],
    'MAE': [],
    'Explained Variance': []
}

for i, column in enumerate(y_train.columns):
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, i], train_predictions_rescaled[:, i]))
    train_r2 = r2_score(y_train.iloc[:, i], train_predictions_rescaled[:, i])
    train_mae = mean_absolute_error(y_train.iloc[:, i], train_predictions_rescaled[:, i])
    train_evar = 1 - np.var(y_train.iloc[:, i] - train_predictions_rescaled[:, i]) / np.var(y_train.iloc[:, i])
    train_metrics['RMSE'].append(train_rmse)
    train_metrics['R²'].append(train_r2)
    train_metrics['MAE'].append(train_mae)
    train_metrics['Explained Variance'].append(train_evar)

    # Validation metrics
    val_rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], val_predictions_rescaled[:, i]))
    val_r2 = r2_score(y_val.iloc[:, i], val_predictions_rescaled[:, i])
    val_mae = mean_absolute_error(y_val.iloc[:, i], val_predictions_rescaled[:, i])
    val_evar = 1 - np.var(y_val.iloc[:, i] - val_predictions_rescaled[:, i]) / np.var(y_val.iloc[:, i])
    val_metrics['RMSE'].append(val_rmse)
    val_metrics['R²'].append(val_r2)
    val_metrics['MAE'].append(val_mae)
    val_metrics['Explained Variance'].append(val_evar)

# Print evaluation metrics for a single output variable
print("Training Metrics:")
print(pd.DataFrame(train_metrics, index=y_train.columns))
print("\nValidation Metrics:")
print(pd.DataFrame(val_metrics, index=y_train.columns))

# Calculate comprehensive evaluation indicators
train_avg_metrics = {
    'Average RMSE': np.mean(train_metrics['RMSE']),
    'Average R²': np.mean(train_metrics['R²']),
    'Average MAE': np.mean(train_metrics['MAE']),
    'Average Explained Variance': np.mean(train_metrics['Explained Variance'])
}
val_avg_metrics = {
    'Average RMSE': np.mean(val_metrics['RMSE']),
    'Average R²': np.mean(val_metrics['R²']),
    'Average MAE': np.mean(val_metrics['MAE']),
    'Average Explained Variance': np.mean(val_metrics['Explained Variance'])
}

# Print comprehensive evaluation indicators
print("\nAverage Training Metrics:")
print(train_avg_metrics)
print("\nAverage Validation Metrics:")
print(val_avg_metrics)


# Draw loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train RMSE')
plt.plot(val_losses, label='Validation RMSE')
plt.xlabel('Number of Boosting Rounds')
plt.ylabel('RMSE')
plt.title('Training and Validation RMSE over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Define the number of target variables
num_targets = y_train.shape[1]

# Draw training results
plt.figure(figsize=(12, 6 * num_targets))  # Dynamically adjust image height based on the number of target variables
for i in range(num_targets):
    plt.subplot(num_targets, 1, i + 1)  # Subgraph index from 1 to num_target
    plt.plot(y_train.iloc[:, i].values, label=f'Actual {y_train.columns[i]}', color='green')
    plt.plot(train_predictions_rescaled[:, i], label=f'Predicted {y_train.columns[i]}', color='orange', alpha=0.7)
    plt.title(f"Training Results for {y_train.columns[i]}")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
plt.tight_layout()
plt.show()

# Draw validation results
plt.figure(figsize=(12, 6 * num_targets))  # Dynamically adjust image height based on the number of target variables
for i in range(num_targets):
    plt.subplot(num_targets, 1, i + 1)  # Subgraph index from 1 to num_target
    plt.plot(y_val.iloc[:, i].values, label=f'Actual {y_val.columns[i]}', color='blue')
    plt.plot(val_predictions_rescaled[:, i], label=f'Predicted {y_val.columns[i]}', color='orange', alpha=0.7)
    plt.title(f"Validation Results for {y_val.columns[i]}")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
plt.tight_layout()
plt.show()

# Save prediction results
train_predictions_rescaled_df = pd.DataFrame(train_predictions_rescaled, columns=y_train.columns)
val_predictions_rescaled_df = pd.DataFrame(val_predictions_rescaled, columns=y_val.columns)

output_file_prefix = "sc-xgb_Predictions"
train_predictions_rescaled_df.to_excel(f'{output_file_prefix}_Train.xlsx', index=False)
val_predictions_rescaled_df.to_excel(f'{output_file_prefix}_Validation.xlsx', index=False)

# Record end time
end_time = time.time()
print(f"Total Runtime: {end_time - start_time:.2f} seconds")
