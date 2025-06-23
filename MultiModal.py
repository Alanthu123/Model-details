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
