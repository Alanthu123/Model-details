import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
import gc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # Correct import for cm module

# Create directory for saving results
results_dir = "TrainResults"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print(f"All results will be saved to: {os.path.abspath(results_dir)}")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalModel, self).__init__()
        
        # Image convolution layers (simplified)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Sequence convolution layers (simplified)
        self.conv_seq1 = nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv_seq2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        
        # Feature dimensionality reduction (to reduce memory usage)
        self.fc_image = nn.Linear(32*32*32, 64)  # Significantly reduce dimension
        self.fc_sequence = nn.Linear(160, 32)     # Sequence feature reduction
        
        # Attention mechanism (dynamic weight generation)
        self.attention_net = nn.Sequential(
            nn.Linear(64+32, 32),  # Reduce layer size
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softplus()  # Ensure weights are positive
        )
        
        # Classification layers
        self.fc1 = nn.Linear(64+32, 32)  # Reduce feature dimension
        self.fc_final = nn.Linear(32, num_classes)
        
    def forward(self, image_data, sequence_data):
        # Image feature extraction
        x = F.relu(self.conv1(image_data))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        image_features = F.relu(self.fc_image(x))
        
        # Sequence feature extraction
        y = sequence_data.permute(0, 2, 1)
        y = F.relu(self.conv_seq1(y))
        y = F.relu(self.conv_seq2(y))
        y = y.view(y.size(0), -1)
        sequence_features = F.relu(self.fc_sequence(y))
        
        # Dynamic weight generation
        combined_features = torch.cat((image_features, sequence_features), dim=1)
        weights = self.attention_net(combined_features)
        w_image = weights[:, 0].unsqueeze(1)
        w_sequence = weights[:, 1].unsqueeze(1)
        
        # Feature fusion with weights
        weighted_image = image_features * w_image
        weighted_sequence = sequence_features * w_sequence
        fused_features = torch.cat((weighted_image, weighted_sequence), dim=1)
        
        # Classification
        x = F.relu(self.fc1(fused_features))
        output = self.fc_final(x)
        
        return output, w_image, w_sequence

def load_data(base_path, sliding_window=10, max_num=2000, img_size=128):  # Reduced image size
    folders = ['Concrete', 'Sand', 'Grass', 'Stone', 'Water']
    sequence_data = []
    labels = []
    image_data = []

    # Iterate through each folder
    for label_index, folder in enumerate(folders, start=0):
        folder_path = os.path.join(base_path, folder)
        num = 0  # Counter

        # Get Excel files
        excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        for excel_file in excel_files:
            excel_path = os.path.join(folder_path, excel_file)
            df = pd.read_excel(excel_path)
            
            # Get number of data rows, exclude header
            total_rows = df.shape[0] - 1
            rows_to_read = total_rows - (total_rows % sliding_window)

            # Limit maximum rows per folder to max_num
            if num + rows_to_read > max_num:
                rows_to_read = max_num - num

            # Read and group data
            for group_index in range(rows_to_read // sliding_window):
                start_row = group_index * sliding_window + 1
                end_row = start_row + sliding_window

                data = df.iloc[start_row:end_row, :4].values
                if data.shape == (sliding_window, 4):
                    sequence_data.append(data)
                    labels.append(label_index)
                    num += 1
                    if num >= max_num:
                        break

            if num >= max_num:
                break

        # Read corresponding images (reduced size)
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        for i in range(num):
            if i < len(image_files):
                image_file = image_files[i]
                image_path = os.path.join(folder_path, image_file)
                image = Image.open(image_path).resize((img_size, img_size))  # Reduce size
                image_array = np.array(image) / 255.0
                image_data.append(image_array.transpose((2, 0, 1)))
                
    # Convert to torch tensors
    sequence_data = torch.tensor(np.array(sequence_data), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    image_data = torch.tensor(np.array(image_data), dtype=torch.float32)
    return image_data, sequence_data, labels

# Load data (using smaller image size)
data_dir = r"todo"
image_data, sequence_data, labels = load_data(data_dir, img_size=128)

# Split dataset
X_train_img, X_test_img, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
    image_data, sequence_data, labels, test_size=0.3, random_state=42, stratify=labels)
X_val_img, X_test_img, X_val_seq, X_test_seq, y_val, y_test = train_test_split(
    X_test_img, X_test_seq, y_test, test_size=0.5, random_state=42, stratify=y_test)

print("Dataset loaded and split completed")

# Create DataLoader (batch loading)
batch_size = 16  # Reduced batch size

train_dataset = TensorDataset(X_train_img, X_train_seq, y_train)
val_dataset = TensorDataset(X_val_img, X_val_seq, y_val)
test_dataset = TensorDataset(X_test_img, X_test_seq, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# GPU memory cleanup
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Model initialization
num_classes = 5
model = MultiModalModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
scaler = torch.amp.GradScaler(device='cuda') # Updated mixed precision API

# Training parameters
num_epochs = 200  # Reduced epochs
best_val_loss = float('inf')
best_model_state = None

# Training logs
train_losses = []
val_losses = []
val_accuracies = []  # New: validation accuracy log
image_weights = []
sequence_weights = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    epoch_img_weights = []
    epoch_seq_weights = []
    
    for images, sequences, labels_batch in train_loader:
        # Move data to GPU
        images = images.float().to(device)
        sequences = sequences.float().to(device)
        labels_batch = labels_batch.long().to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training (reduce GPU memory) - updated API
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs, w_img, w_seq = model(images, sequences)
            loss = criterion(outputs, labels_batch)
        
        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        epoch_img_weights.append(w_img.mean().item())
        epoch_seq_weights.append(w_seq.mean().item())
        
        # Release memory
        del images, sequences, labels_batch, outputs, w_img, w_seq
        clear_memory()
    
    # Record average training loss and weights per epoch
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    image_weights.append(np.mean(epoch_img_weights))
    sequence_weights.append(np.mean(epoch_seq_weights))
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, sequences, labels_batch in val_loader:
            images = images.float().to(device)
            sequences = sequences.float().to(device)
            labels_batch = labels_batch.long().to(device)
            
            outputs, _, _ = model(images, sequences)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            val_total += labels_batch.size(0)
            val_correct += (predicted == labels_batch).sum().item()
            
            # Release memory
            del images, sequences, labels_batch, outputs
            clear_memory()
            
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = val_correct / val_total
    val_accuracies.append(val_accuracy)  # Record validation accuracy
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}%')

# Plot training metrics separately and save automatically
# 1. Loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
plt.close()  # Close figure to prevent overlap

# 2. Weight adjustment curves
plt.figure(figsize=(10, 6))
plt.plot(image_weights, label='Image Weight')
plt.plot(sequence_weights, label='Sequence Weight')
plt.xlabel('Epochs')
plt.ylabel('Weight Value')
plt.title('Dynamic Weight Adjustment')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'weight_adjustment.png'))
plt.close()  # Close figure

# 3. Accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'accuracy_curve.png'))
plt.close()  # Close figure

print("Training metric plots saved")

# Save best model
torch.save(best_model_state, os.path.join(results_dir, 'best_dynamic_model.pth'))

# Testing phase
model.eval()
test_preds = []
test_labels = []
img_weights_test = []
seq_weights_test = []

with torch.no_grad():
    model.load_state_dict(best_model_state)
    
    for images, sequences, labels_batch in test_loader:
        images = images.float().to(device)
        sequences = sequences.float().to(device)
        labels_batch = labels_batch.long().to(device)
        
        outputs, w_img, w_seq = model(images, sequences)
        _, preds = torch.max(outputs, 1)
        
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels_batch.cpu().numpy())
        img_weights_test.append(w_img.mean().item())
        seq_weights_test.append(w_seq.mean().item())
        
        # Release memory
        del images, sequences, labels_batch, outputs, w_img, w_seq
        clear_memory()

# Calculate final weights
final_image_weight = np.mean(img_weights_test)
final_sequence_weight = np.mean(seq_weights_test)

# Calculate accuracy
accuracy = np.mean(np.array(test_preds) == np.array(test_labels))

# Print optimal weights
print(f'\nOptimal Weights after Training:')
print(f'Image Weight: {final_image_weight:.4f}')
print(f'Sequence Weight: {final_sequence_weight:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Confusion matrix
conf_mat = confusion_matrix(test_labels, test_preds)  # Variable renamed to avoid conflict with cm module
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Concrete', 'Sand', 'Grass', 'Stone', 'Water'],
            yticklabels=['Concrete', 'Sand', 'Grass', 'Stone', 'Water'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'normalized_confusion_matrix.png'))
plt.close()  # Close figure

# Save weights and results
result_df = pd.DataFrame({
    'Metric': ['Image Weight', 'Sequence Weight', 'Accuracy'],
    'Value': [final_image_weight, final_sequence_weight, accuracy]
})
result_df.to_excel(os.path.join(results_dir, 'dynamic_weights_results.xlsx'), index=False)

# =============================================================================
# New feature: Plot accuracy vs. weight combinations (weights sum to 1)
# =============================================================================

print("\nStarting weight-accuracy surface analysis (weights sum to 1)...")

# Use trained feature extraction components
class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        # Copy image feature extraction layers
        self.conv1 = base_model.conv1
        self.conv2 = base_model.conv2
        self.pool = base_model.pool
        self.fc_image = base_model.fc_image
        
        # Copy sequence feature extraction layers
        self.conv_seq1 = base_model.conv_seq1
        self.conv_seq2 = base_model.conv_seq2
        self.fc_sequence = base_model.fc_sequence
        
        # Copy classification layers
        self.fc1 = base_model.fc1
        self.fc_final = base_model.fc_final
        
    def forward(self, image_data, sequence_data, img_weight, seq_weight):
        # Image feature extraction
        x = F.relu(self.conv1(image_data))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        image_features = F.relu(self.fc_image(x))
        
        # Sequence feature extraction
        y = sequence_data.permute(0, 2, 1)
        y = F.relu(self.conv_seq1(y))
        y = F.relu(self.conv_seq2(y))
        y = y.view(y.size(0), -1)
        sequence_features = F.relu(self.fc_sequence(y))
        
        # Weighted feature fusion
        weighted_image = image_features * img_weight
        weighted_sequence = sequence_features * seq_weight
        fused_features = torch.cat((weighted_image, weighted_sequence), dim=1)
        
        # Classification
        x = F.relu(self.fc1(fused_features))
        output = self.fc_final(x)
        
        return output

# Create feature extractor
feature_extractor = FeatureExtractor(model).to(device)
feature_extractor.eval()

# Prepare grid data (weight range 0-1, step 0.05)
weight_range = np.linspace(0, 1, 21)  # 0 to 1 with step 0.05
accuracy_grid = np.zeros(len(weight_range))

# Use validation set for testing
val_loader_small = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Iterate through all weight combinations (image weight varies, sequence weight = 1 - image weight)
for i, img_w in enumerate(weight_range):
    seq_w = 1 - img_w  # Ensure weights sum to 1
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, sequences, labels_batch in val_loader_small:
            images = images.float().to(device)
            sequences = sequences.float().to(device)
            labels_batch = labels_batch.long().to(device)
            
            # Convert to scalar tensors
            img_weight_tensor = torch.tensor(img_w, dtype=torch.float32).to(device)
            seq_weight_tensor = torch.tensor(seq_w, dtype=torch.float32).to(device)
            
            outputs = feature_extractor(images, sequences, img_weight_tensor, seq_weight_tensor)
            _, predicted = torch.max(outputs, 1)
            
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    
    accuracy = correct / total
    accuracy_grid[i] = accuracy
    print(f'Img Weight: {img_w:.2f}, Seq Weight: {seq_w:.2f}, Accuracy: {accuracy*100:.2f}%')

# Save accuracy data
accuracy_df = pd.DataFrame({
    'Image_Weight': weight_range,
    'Sequence_Weight': 1 - weight_range,
    'Accuracy': accuracy_grid
})
accuracy_df.to_excel(os.path.join(results_dir, 'accuracy_weight_sum1.xlsx'), index=False)

# Plot weight-accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(weight_range, accuracy_grid, 'b-', linewidth=2)
plt.xlabel('Image Weight (Sequence Weight = 1 - Image Weight)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Weight Combination (Sum=1)')

# Mark highest accuracy point
max_idx = np.argmax(accuracy_grid)
plt.scatter(weight_range[max_idx], accuracy_grid[max_idx], 
            c='red', s=100, marker='*', 
            label=f'Max Acc: {accuracy_grid[max_idx]*100:.2f}%')

# Calculate normalized training weights
total_weight = final_image_weight + final_sequence_weight
norm_img_weight = final_image_weight / total_weight
norm_seq_weight = final_sequence_weight / total_weight

# Find accuracy at training weights
train_acc_index = np.abs(weight_range - norm_img_weight).argmin()
train_acc = accuracy_grid[train_acc_index]

# Mark training weights point
plt.scatter(norm_img_weight, train_acc, 
            c='green', s=100, marker='o', 
            label=f'Trained Weights (Acc: {train_acc*100:.2f}%)')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'accuracy_weight_curve_sum1.png'))
plt.close()

print("Weight-accuracy analysis completed (weights sum to 1)!")
print(f"All results saved to: {os.path.abspath(results_dir)}")
