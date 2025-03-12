import torch
import torch.nn as nn

class UNet2D(nn.Module):
    """
    2D version of the classification UNet architecture.
    
    Args:
        in_channels (int): Number of input channels (default: 1)
        init_features (int): Number of features in first layer (default: 32)
        dropout_prob (float): Dropout probability in classification head (default: 0.5)
    """
    def __init__(self, in_channels=1, init_features=32, dropout_prob=0.5):
        super(UNet2D, self).__init__()

        # Store feature numbers for each level
        features = init_features
        
        # Level 1 (No pooling)  
        num_groups = 1 if in_channels == 1 else 8 # Adjust the number of groups to 1 if in_channels is 1
        self.level1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Level 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.level2 = nn.Sequential(
            nn.GroupNorm(8, features),
            nn.Conv2d(features, features*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Level 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.level3 = nn.Sequential(
            nn.GroupNorm(8, features*2),
            nn.Conv2d(features*2, features*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Level 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.level4 = nn.Sequential(
            nn.GroupNorm(8, features*4),
            nn.Conv2d(features*4, features*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(features*8, 1)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, 1)
        
        Shape transformations:
        Level 1: (batch, 1, H, W) -> (batch, 32, H, W)
        Level 2: (batch, 32, H/2, W/2) -> (batch, 64, H/2, W/2)
        Level 3: (batch, 64, H/4, W/4) -> (batch, 128, H/4, W/4)
        Level 4: (batch, 128, H/8, W/8) -> (batch, 256, H/8, W/8)
        Classification: (batch, 256, H/8, W/8) -> (batch, 1)
        """
        # Encoder path
        x1 = self.level1(x)           # (batch, 32, H, W)
        
        x2 = self.pool2(x1)           # (batch, 32, H/2, W/2)
        x2 = self.level2(x2)          # (batch, 64, H/2, W/2)
        
        x3 = self.pool3(x2)           # (batch, 64, H/4, W/4)
        x3 = self.level3(x3)          # (batch, 128, H/4, W/4)
        
        x4 = self.pool4(x3)           # (batch, 128, H/8, W/8)
        x4 = self.level4(x4)          # (batch, 256, H/8, W/8)
        
        # Classification head
        out = self.classifier(x4)      # (batch, 1)
        
        return out

    def get_embedding(self, x):
        # Encoder path
        x1 = self.level1(x)           # (batch, 32, H, W)
        
        x2 = self.pool2(x1)           # (batch, 32, H/2, W/2)
        x2 = self.level2(x2)          # (batch, 64, H/2, W/2)
        
        x3 = self.pool3(x2)           # (batch, 64, H/4, W/4)
        x3 = self.level3(x3)          # (batch, 128, H/4, W/4)
        
        x4 = self.pool4(x3)           # (batch, 128, H/8, W/8)
        x4 = self.level4(x4)          # (batch, 256, H/8, W/8)

        # Classification head (w/o final linear layer)
        pooled = self.classifier[0](x4)
        embedding = self.classifier[1](pooled)

        return embedding

if __name__ == "__main__":
    # Test the model with a sample input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D().to(device)
    
    # Create random input tensor (batch_size=4, channels=1, height=64, width=64)
    x = torch.randn(4, 1, 64, 64).to(device)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print model summary
    print("\nModel Architecture:")
    print(model) 