Fish Freshness Analysis – MATLAB Project

Overview

This project uses image processing and machine learning in MATLAB to automatically classify fish images into fresh, moderate, or spoiled categories. It features a simple GUI for training and testing the model without coding knowledge.

Features
	•	GUI-based Interface – Users can select actions from buttons:
	1.	Train Model – Train a classification model on labeled fish images.
	2.	Predict Single Image – Load an image and predict its freshness category.
	3.	Exit – Close the application.
	•	Image Feature Extraction – Extracts relevant visual features from fish images (color, texture, etc.).
	•	Machine Learning Classifier – Trains a model using the extracted features and user-labeled datasets.
	•	Prediction Output – Displays predicted freshness with confidence.

How It Works
	1.	Start the Application
Run the fishFreshnessML() function in MATLAB. The GUI window will appear.
	2.	Training the Model
	•	Choose the Train Model button.
	•	Select a folder containing three subfolders:
	•	fresh – images of fresh fish
	•	moderate – images of moderately fresh fish
	•	spoiled – images of spoiled fish
	•	The script processes each image, extracts features, and trains a classifier.
	3.	Predicting Freshness
	•	Choose Predict Single Image.
	•	Load an image of a fish.
	•	The program will classify it as fresh, moderate, or spoiled.
	4.	Exit
	•	Click the Exit button to close the application.

Requirements
	•	MATLAB (with Image Processing Toolbox & Statistics and Machine Learning Toolbox)
	•	A labeled dataset of fish images.

Example Folder Structure for Training

dataset/
│
├── fresh/
│   ├── fish1.jpg
│   ├── fish2.jpg
│
├── moderate/
│   ├── fish3.jpg
│   ├── fish4.jpg
│
└── spoiled/
    ├── fish5.jpg
    ├── fish6.jpg

Usage

fishFreshnessML()

Output
	•	Displays a GUI for model training and prediction.
	•	Shows classification result for a selected image with the predicted category.


