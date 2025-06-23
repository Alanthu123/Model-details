import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import precision_recall_curve

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MultiModalModel(nn.Module):
    def __init__(self, num_classes, image_weight=1.0, sequence_weight=0.2):
        super(MultiModalModel, self).__init__()
        
        # CNN layers for image data
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # CNN layers for sequence data
        self.conv_seq1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)  # Reduced output channels
        self.conv_seq2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)  # Reduced output channels

        # Final classification layer
        self.fc1 = None  # Initialized later based on input size
        self.fc_final = nn.Linear(64, num_classes)  # Input features: 64
        
        # Feature weighting parameters
        self.image_weight = image_weight
        self.sequence_weight = sequence_weight

    def forward(self, image_data, sequence_data):
        # Image data forward pass
        x = F.relu(self.conv1(image_data))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        image_features = x * self.image_weight  # Apply image weight

        # Sequence data forward pass
        y = sequence_data.permute(0, 2, 1)  # Reshape to (batch, channels, seq_length)
        y = F.relu(self.conv_seq1(y))
        y = F.relu(self.conv_seq2(y))
        y = y.view(y.size(0), -1)  # Flatten
        sequence_features = y * self.sequence_weight  # Apply sequence weight

        # Concatenate features
        combined_features = torch.cat((image_features, sequence_features), dim=1)
        
        # Dynamically initialize fc1 if needed
        if self.fc1 is None:
            self.fc1 = nn.Linear(combined_features.size(1), 64).to(device)
        
        fc1 = F.relu(self.fc1(combined_features))
        output = self.fc_final(fc1)
        return output

def load_data(base_path, sliding_window=10, max_num=2000):
    folders = ['Concrete', 'Sand', 'Grass', 'Stone', 'Water']
    sequence_data = []  # Store sequence data
    labels = []        # Store labels
    image_data = []    # Store image data

    # Process each category folder
    for label_index, folder in enumerate(folders, start=0):
        folder_path = os.path.join(base_path, folder)
        num = 0  # Counter for current category

        # Process Excel files
        excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        for excel_file in excel_files:
            excel_path = os.path.join(folder_path, excel_file)
            df = pd.read_excel(excel_path)  # Read Excel file

            # Calculate readable rows (multiples of sliding_window)
            total_rows = df.shape[0] - 1  # Exclude header
            rows_to_read = total_rows - (total_rows % sliding_window)

            # Enforce max_num limit
            if num + rows_to_read > max_num:
                rows_to_read = max_num - num

            # Process data in sliding_window chunks
            for group_index in range(rows_to_read // sliding_window):
                start_row = group_index * sliding_window + 1  # Start after header
                end_row = start_row + sliding_window
                
                # Extract first 4 columns
                data = df.iloc[start_row:end_row, :4].values
                if data.shape == (sliding_window, 4):
                    sequence_data.append(data)
                    labels.append(label_index)
                    num += 1
                    if num >= max_num:
                        break
            if num >= max_num:
                break

        # Load corresponding images
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        for i in range(num):
            if i < len(image_files):
                image_path = os.path.join(folder_path, image_files[i])
                image = Image.open(image_path).resize((256, 256))  # Resize
                image_array = np.array(image) / 255.0  # Normalize
                image_data.append(image_array.transpose((2, 0, 1)))  # Convert to CxHxW

    # Convert to PyTorch tensors
    sequence_data = torch.tensor(np.array(sequence_data), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    image_data = torch.tensor(np.array(image_data), dtype=torch.float32)
    return image_data, sequence_data, labels

# Main execution
if __name__ == "__main__":
    data_dir = r"E:\1科研仿生龟\19TRO投稿\4乌龟实验数据\241024数据3"
    image_data, sequence_data, labels = load_data(data_dir)
    
    # Split dataset (60% train, 20% validation, 20% test)
    X_train_img, X_test_img, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        image_data, sequence_data, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val_img, X_test_img, X_val_seq, X_test_seq, y_val, y_test = train_test_split(
        X_test_img, X_test_seq, y_test, test_size=0.5, random_state=42, stratify=y_test)

    print("Dataset loaded and split successfully")
    print(f"Training samples: {len(X_train_img)}")
    print(f"Validation samples: {len(X_val_img)}")
    print(f"Testing samples: {len(X_test_img)}")

    # Initialize model
    num_classes = 5
    model = MultiModalModel(num_classes, image_weight=1.5, sequence_weight=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    # Training parameters
    num_epochs = 800
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Move data to device
        X_train_img, X_train_seq, y_train = X_train_img.to(device), X_train_seq.to(device), y_train.to(device)
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(X_train_img, X_train_seq)
            loss = criterion(outputs, y_train)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_img, X_val_seq, y_val = X_val_img.to(device), X_val_seq.to(device), y_val.to(device)
            val_outputs = model(X_val_img, X_val_seq)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
            _, val_preds = torch.max(val_outputs, 1)
            print(f'Validation Loss: {val_loss.item():.4f}')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

    # Save best model
    torch.save(best_model_state, 'best_model.pth')
    print("Best model saved to best_model.pth")

    # Testing
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        X_test_img, X_test_seq, y_test = X_test_img.to(device), X_test_seq.to(device), y_test.to(device)
        test_outputs = model(X_test_img, X_test_seq)
        _, test_preds = torch.max(test_outputs, 1)

        # Generate confusion matrix
        cm = confusion_matrix(y_test.cpu(), test_preds.cpu())
        print("Confusion Matrix:\n", cm)

        # Save confusion matrix
        cm_df = pd.DataFrame(cm, 
                            index=['Concrete', 'Sand', 'Grass', 'Stone', 'Water'],
                            columns=['Concrete', 'Sand', 'Grass', 'Stone', 'Water'])
        cm_df.to_excel('confusion_matrix.xlsx')

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()

        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')
        plt.savefig('confusion_matrix_normalized.png')
        plt.show()
