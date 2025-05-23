import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor, DistilBertModel, DistilBertTokenizer

# Define the Tokenizer and ImageProcessor

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', clean_up_tokenization_spaces=True)
transform = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# The ImageEncoder class uses a Vision Transformer (ViT) model to extract features from images and projects them to a size of 256 dimensions.
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Load the Vision Transformer (ViT) model
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = nn.Linear(self.vit.config.hidden_size, 256)

    def forward(self, images):
        # Extract features from the ViT model
        outputs = self.vit(images)
        features = outputs.last_hidden_state[:, 0, :]
        return self.fc(features)


# The TextEncoder class utilizes a DistilBERT model to encode text inputs and projects the output to 256 dimensions for further processing.
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        # Load the DistilBERT model for text encoding
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Linear(768, 256)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(output.last_hidden_state[:, 0, :])


# The ProjectionHead class is designed to refine and project input embeddings into a specified dimensional space.
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=256, projection_dim=256, dropout=0.1):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x) # Second fully connected layer
        x = self.dropout(x)
        x = x + projected # Residual connection (Skip connection)
        x = self.layer_norm(x)
        return x


# The CLIPModel class combines an image encoder, a text encoder, and a projection head to create a model that extracts features from both images and text,
# then projects them into a common embedding space.
class CLIPModel(nn.Module):
    def __init__(self):
        super(CLIPModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.projection = ProjectionHead()

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        image_projection = self.projection(image_features)
        text_projection = self.projection(text_features)
        return image_projection, text_projection

from google.colab import drive
drive.mount('/content/drive')
