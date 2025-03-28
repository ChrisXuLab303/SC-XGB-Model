import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Record start time
start_time = time.time()

# Step 1: Load and preprocess the data
train_data = pd.read_excel('E:\\2_MulObjRules\\ML_whl\\BP_whl_2025\\input2_BP_TrainFit.xlsx')
validation_data = pd.read_excel('E:\\2_MulObjRules\\ML_whl\\BP_whl_2025\\input2_BP_ValidationTest.xlsx')



# Data partitioning
X_train = train_data.iloc[:, :-4].values.astype(np.float32)
y_train = train_data.iloc[:, -4:].values.astype(np.float32)
X_val = validation_data.iloc[:, :-4].values.astype(np.float32)
y_val = validation_data.iloc[:, -4:].values.astype(np.float32)

# View statistical information
print("Training Target Statistics:")
print(pd.DataFrame(y_train).describe())
print("\nValidation Target Statistics:")
print(pd.DataFrame(y_val).describe())

# Step 2: Define the BP Neural Network Model
class BPNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(BPNet, self).__init__()
        self.hidden1 = torch.nn.Linear(in_features=input_size, out_features=128)
        self.hidden2 = torch.nn.Linear(in_features=64, out_features=32)
        self.output = torch.nn.Linear(in_features=32, out_features=output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

# Custom loss function: MSE+negative penalty
class CustomLoss(nn.Module):
    def __init__(self, penalty_weight=10.0):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)  # Original MSE loss
        negative_penalty = torch.sum(torch.clamp(predictions, max=0)**2)  # Negative value penalty item
        total_loss = mse_loss + self.penalty_weight * negative_penalty
        return total_loss

# Initialize the model
input_size = X_train.shape[1]
output_size = y_train.shape[1]
model = BPNet(input_size, output_size)
print(model)

# Step 3: Define loss function and optimizer
criterion = CustomLoss(penalty_weight=10.0)  # Custom loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the model
num_epochs = 100
batch_size = 32

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)  # Use custom loss function
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


# Step 5: Save and visualize results
torch.save(model.state_dict(), 'bp_model.pth')

# Draw loss curve
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Step 6: Evaluate the model
model.eval()
with torch.no_grad():
    y_train_pred = model(torch.tensor(X_train)).numpy()
    y_val_pred = model(torch.tensor(X_val)).numpy()

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

for i in range(y_train.shape[1]):
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train[:, i], y_train_pred[:, i]))
    train_r2 = r2_score(y_train[:, i], y_train_pred[:, i])
    train_mae = mean_absolute_error(y_train[:, i], y_train_pred[:, i])
    train_evar = 1 - np.var(y_train[:, i] - y_train_pred[:, i]) / np.var(y_train[:, i])
    train_metrics['RMSE'].append(train_rmse)
    train_metrics['R²'].append(train_r2)
    train_metrics['MAE'].append(train_mae)
    train_metrics['Explained Variance'].append(train_evar)

    # Validation metrics
    val_rmse = np.sqrt(mean_squared_error(y_val[:, i], y_val_pred[:, i]))
    val_r2 = r2_score(y_val[:, i], y_val_pred[:, i])
    val_mae = mean_absolute_error(y_val[:, i], y_val_pred[:, i])
    val_evar = 1 - np.var(y_val[:, i] - y_val_pred[:, i]) / np.var(y_val[:, i])
    val_metrics['RMSE'].append(val_rmse)
    val_metrics['R²'].append(val_r2)
    val_metrics['MAE'].append(val_mae)
    val_metrics['Explained Variance'].append(val_evar)

# Print evaluation metrics for a single output variable
print("Training Metrics:")
print(pd.DataFrame(train_metrics))
print("\nValidation Metrics:")
print(pd.DataFrame(val_metrics))

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


# Step 7: Model Evaluation and Visualization
model.eval()
with torch.no_grad():
    y_train_pred = model(torch.tensor(X_train)).numpy()
    y_val_pred = model(torch.tensor(X_val)).numpy()

# Visualize training and validation results
num_targets = y_train.shape[1]  # Number of target variables

# Draw training results
plt.figure(figsize=(12, 6 * num_targets))
for i in range(num_targets):
    plt.subplot(num_targets, 1, i + 1)
    plt.plot(y_train[:, i], label=f'Actual Target {i+1}', color='green')
    plt.plot(y_train_pred[:, i], label=f'Predicted Target {i+1}', color='orange', alpha=0.7)
    plt.title(f"Training Results for Target {i+1}")
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
plt.tight_layout()
plt.show()

# Draw validation results
plt.figure(figsize=(12, 6 * num_targets))
for i in range(num_targets):
    plt.subplot(num_targets, 1, i + 1)
    plt.plot(y_val[:, i], label=f'Actual Target {i+1}', color='blue')
    plt.plot(y_val_pred[:, i], label=f'Predicted Target {i+1}', color='orange', alpha=0.7)
    plt.title(f"Validation Results for Target {i+1}")
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
plt.tight_layout()
plt.show()

# Save prediction results
train_predictions_df = pd.DataFrame(y_train_pred, columns=[f"Target_{i+1}" for i in range(y_train.shape[1])])
val_predictions_df = pd.DataFrame(y_val_pred, columns=[f"Target_{i+1}" for i in range(y_val.shape[1])])

output_file_prefix = "BPNet_Predictions"
train_predictions_df.to_excel(f'{output_file_prefix}_Train.xlsx', index=False)
val_predictions_df.to_excel(f'{output_file_prefix}_Validation.xlsx', index=False)

# Total time
end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")

