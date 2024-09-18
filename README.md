# Multi-Digit Image Detection using RNN and Sliding Window Techniques

This repository contains code and resources for detecting multiple digits from images using deep learning techniques. The project is based on a pre-trained MNIST model and utilizes a sliding window approach to detect multiple digits in images with varying widths. It also features image preprocessing steps, prediction functions, and result visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Benefits and Achievements](#benefits-and-achievements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Preprocessing Images](#preprocessing-images)
  - [Sliding Window Prediction](#sliding-window-prediction)
- [Steps for Detection](#steps-for-detection)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to detect multiple digits from grayscale images with non-standard widths (but standard heights). It applies a sliding window technique to extract 28x28 patches across the width of an image and runs predictions using a pre-trained Convolutional Neural Network (CNN) based on the MNIST dataset.

## Features

- **Pre-trained MNIST Model**: Leverages a pre-trained model trained on the MNIST dataset to recognize digits from 0 to 9.
- **Sliding Window Technique**: Dynamically applies a 28x28 window to scan the image horizontally, allowing the detection of multiple digits.
- **Customizable Parameters**: Adjust window step size and prediction thresholds to tune performance.
- **Image Preprocessing**: Normalizes grayscale images and handles images of various widths without resizing the width.
- **Efficient Predictions**: Minimizes unnecessary predictions by stopping the scan when appropriate conditions are met.

## Benefits and Achievements

- **Accurate Multi-Digit Detection**: The model can accurately detect and classify multiple digits in images of varying widths.
- **Flexible Input Handling**: Designed to handle grayscale images with any width as long as the height is fixed at 28 pixels.
- **Scalable**: The approach can be extended to larger datasets or more complex digit recognition tasks.
- **Real-World Applications**: Suitable for document scanning, automated number recognition (e.g., license plates, postal codes), and digit recognition in noisy environments.

## Setup and Installation

To get started with the project, follow the steps below:

### Prerequisites
- Python 3.x
- Jupyter Notebook or any IDE of your choice
- TensorFlow and Keras
- `tqdm` for progress tracking
- `cv2` for image processing (OpenCV)

### Install Required Libraries

```bash
pip install tensorflow keras tqdm opencv-python
```

### Clone the Repository

```bash
git clone https://github.com/amirrgb/Multi-Digit-Image-Detection-using-RNN.git
cd Multi-Digit-Image-Detection-using-RNN
```

## Usage

### Preprocessing Images

You can preprocess your images before running the sliding window prediction. Ensure your images are grayscale and have a height of 28 pixels. If needed, the script can normalize the pixel values.

### Sliding Window Prediction

The main function to detect multiple digits in an image is `predict_multi_digits_sliding_window`. Here’s a sample usage:

```python
detected_digits = predict_multi_digits_sliding_window(model, image_path, new_dir_path)
```

- `model`: The pre-trained CNN model.
- `image_path`: Path to the input image.
- `new_dir_path`: Directory to save output images with detected digits marked.

The function returns a list of detected digits.

### Example Workflow

1. **Load the Model**: Load the pre-trained MNIST model using Keras.
2. **Preprocess Images**: Ensure the input images have a height of 28 pixels.
3. **Run Predictions**: Apply the sliding window to detect digits and store the results.

## Steps for Detection

1. **Image Loading**: Read the image using OpenCV and ensure it's a grayscale image.
2. **Sliding Window**: Starting from the leftmost column, a 28x28 window slides across the width of the image one column at a time.
3. **Prediction**: For each window, the model predicts the digit (if any) with a probability threshold.
4. **Digit Storing**: Detected digits are stored and saved as part of the result.
5. **Repeat**: The sliding window moves by 1 column and repeats until the end of the image is reached.

## Model Details

- **Architecture**: A CNN model with two convolutional layers, followed by max pooling, flattening, and fully connected layers. The model is trained on the MNIST dataset for recognizing digits.
- **Input**: The model expects 28x28 grayscale images as input.
- **Output**: The model outputs probabilities for each of the 10 digits (0-9).

### Model Code Snippet

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Results

- The model is capable of identifying digits with high accuracy.
- Predictions are stored in CSV format for further analysis.
- Detected digits are visualized by marking bounding boxes on the original image.

### Example of Detected Digits

```python
print("Detected Digits:", detected_digits)
```

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and test them.
4. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
