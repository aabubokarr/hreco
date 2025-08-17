# CSE445-Project-MLD

This repository contains the implementation of the Machine Learning-based project for CSE445.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project focuses on leveraging machine learning techniques to solve the communication gap between those who don't know sign language.  
The aim is to translate sign language into text (and optionally speech) using a trained deep learning model.

## Features
- Hand Landmark Extraction using MediaPipe  
- ASL Hand Sign Recognition using TensorFlow/Keras  
- Image-based Prediction (`test.py`)  
- Real-time Prediction via Webcam (`main.py`)  
- Flask-based Web Application (`app.py`)

## Technologies Used
- Python  
- Flask  
- TensorFlow / Keras  
- OpenCV  
- MediaPipe  
- scikit-learn  
- Numpy  

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/imabu0/CSE445-Project-MLD.git
    ```
2. Navigate to the project directory:
    ```bash
    cd CSE445-Project-MLD
    ```
3. Install dependencies:
    ```bash
    pip install flask scikit-learn opencv-python mediapipe tensorflow numpy
    ```

## Usage
1. **Train the Model**  
    Run the training script to generate the model (`asl.h5`):
    ```bash
    python train.py
    ```

2. **Test with a Single Image**  
    Run the test script with an input image:
    ```bash
    python test.py path/to/sample_image.png
    ```

3. **Run Real-time Recognition**  
    Use your webcam for live predictions:
    ```bash
    python main.py
    ```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.  
2. Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Add feature-name"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-name
    ```
5. Open a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
