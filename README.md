---
# CLIP-Foundation  

## Project Description  
This project is focused on designing a **CLIP (Contrastive Language-Image Pretraining) model from scratch** to gain a deep understanding of its architecture and functionality. The primary goal is to **learn how to build and implement a CLIP model** while acquiring experience in embedding text and images into a shared representation space.  

To solidify this learning process, the model has been trained on an **image-text matching task**, demonstrating its ability to align textual descriptions with corresponding visual representations.  

## Overall Goal of the Project  
The main objective of this project is to **develop a strong foundation in CLIP architecture and text-image embedding techniques**. The knowledge and experience gained from this implementation will be valuable for future **text-to-image (T2I) and text-to-video (T2V) generative AI projects**, helping to understand how text embedding should be handled effectively in such models.  

### Dataset Used
The **Flickr30k** dataset, licensed under **CC0: Public Domain**, was used in this project. This dataset is suitable and widely used for tasks such as text-based image matching, image captioning, and similar applications. The dataset can be freely used in both commercial and non-commercial projects without any restrictions.

For more information, visit the [Flickr30k dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset) page.

---

## Technical Details and Architecture

This project merges text and visual data into a shared embedding space for text-image matching and advanced future projects. The architecture used in the project includes two separate encoders for text and image processing, along with a projection head, based on the principles of the CLIP model.

#### Model Components
- **ImageEncoder Class**: Extracts visual features using a Vision Transformer (ViT) model. The visual features are projected to a 256-dimensional output after passing through the ViT model.
  - **ViTModel**: Used for feature extraction from images and has a transformer-based structure.
  - **Fully Connected Layer (fc)**: Projects the extracted features from the ViT to 256 dimensions.
  - **Forward Function**: Processes images through the ViT model, extracts features from `last_hidden_state`, and passes them through the projection layer.

- **TextEncoder Class**: Encodes text inputs using a DistilBERT model and projects the output to a 256-dimensional vector.
  - **DistilBERTModel**: A lightweight and fast transformer model for processing text data.
  - **Fully Connected Layer (fc)**: Reduces the text embedding output to 256 dimensions.
  - **Forward Function**: Processes input texts through the BERT model, extracts the first element from `last_hidden_state`, and passes it through the projection layer.

- **ProjectionHead Class**: Ensures higher quality and consistent projection of visual and text embeddings into a shared space. This layer projects the input vectors into a 256-dimensional space using residual connections.
  - **Linear Layer**: Projects the input embedding.
  - **GELU Activation Function**: Used as a non-linear activation function.
  - **Dropout and Layer Norm**: Used to improve generalization and prevent overfitting.

- **CLIPModel Class**: Combines visual and text encoders and projects both data types into a shared embedding space.
  - **Forward Function**: Passes visual and text inputs through their respective encoders and uses the projection head to merge the two embedding spaces. The projections are used for similarity calculations between image and text features.

### Technical Specifications
- **Models Used**:
  - For images: Vision Transformer (ViT)
  - For text: DistilBERT
- **Projection Layer**:
  - Dimension: 256
  - Layers: Two fully connected layers, dropout, layer norm, and residual connections.
- **Activation Functions**:
  - **GELU (Gaussian Error Linear Unit)**: Used as the non-linear activation function in the projection layer.
- **Input Format**:
  - Images: Converted to tensor format using `transform` operations in RGB format.
  - Texts: Tokenized and converted to tensor format with a maximum length of 64.

This structure enables the project to effectively process and evaluate text and visual data within a shared space and provides a flexible architecture for future enhancements.

---

## Data Preprocessing and Transformations

This project includes data preprocessing and transformation steps to prepare text and image data for the model.
### Image Data Preprocessing
The image data is prepared to be compatible with the Vision Transformer (ViT) model. The steps for image processing include:
- **Image Loading and Color Conversion**: Images are loaded using the `PIL` library and converted to RGB format.
- **Transformations**: The image data is processed with the pre-trained transformation operations (`transform`) defined in the `clipModel.py` file. This transformation normalizes pixel values and converts images into the appropriate tensor format for the model.
- **Extracting Pixel Values**: The operation `transform(images=image)['pixel_values'][0]` converts the image pixels into the suitable tensor structure for the model input.

These steps optimize the images to meet the input requirements of the ViT model, enhancing model accuracy.

#### Text Data Preprocessing
The text data is prepared to be compatible with the DistilBERT-based text encoder. The text preprocessing steps include:
- **Tokenization**: The `tokenizer` in the `clipModel.py` file tokenizes the text inputs into a BERT-based format.
- **Padding and Truncation**: Texts are processed with `padding='max_length'` and `truncation=True` to ensure a maximum length of 64. This step keeps text lengths uniform, optimizing model performance.
- **Conversion to Tensor Format**: The tokenized data is converted to PyTorch tensors with `return_tensors='pt'`, making it compatible with the model input format.

---

## Project Structure and Modularity
The project is modular and designed for future expansion. The main model architecture is defined in the `clipModel.py` file. This file includes the tokenizer and transformation processes used for both text and visual data, providing a solid foundation for the integration of GAN models in the future.

### Main Files:
- **build_model.ipynb**: The primary Jupyter Notebook file for training, evaluating, and visualizing text-image matching results.
- **clipModel.py**: A modular file that includes the text and image encoders, projection heads, tokenizer, and transformation processes.

## Usage of the Project
You can use this project to detect the best-matching images based on text inputs. Installation and usage instructions are provided below:

### Required Technologies and Libraries
The following libraries are required to run the project:
- **Python 3.10+**
- **PyTorch (with CUDA support)**
- **Transformers (Hugging Face)**
- **Torchvision**
- **scikit-learn**
- **Pandas**
- **PIL (Pillow)**
- **Matplotlib**
- **TQDM**

### Installation Steps
1. **Install the required libraries**:
    ```bash
    pip install torch torchvision transformers scikit-learn pandas matplotlib pillow tqdm
    ```

2. **Install CUDA-enabled PyTorch** (for GPU support):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

3. **Download and prepare the Flickr30k dataset**:
   - This project uses the Flickr30k dataset. Use your `kaggle` API key to download the dataset with the following commands:
     ```bash
     mkdir -p ~/.kaggle
     mv kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     kaggle datasets download -d hsankesara/flickr-image-dataset
     unzip flickr-image-dataset.zip -d flickr30k
     ```

4. **Include the model files and other code in your project**:
   Ensure the `clipModel.py` file and other necessary code files are in your project directory.

### Training and Using the Model
1. Open `build_model.ipynb` to train and evaluate the model. Run the cells in order.
2. Use the `train()` function to train the model and the `evaluate()` function to evaluate it.
3. Save the trained model output to the path specified in the `model_path` variable. The `build_model.ipynb` file is coded for the Colab environment.
4. Use the `find_matches()` function to find the best-matching images for a given text input.

### Example Usage:
```python
find_matches("dog")
find_matches("plane")
find_matches("Gloomy weather")
find_matches("someone dunks on the basket")
```
## License
This project is licensed under the [MIT License](LICENSE). For more details, please review the license file.
