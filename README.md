<p align="center">
	<img src="https://badgen.net/badge/TensorFlow/2.11.0/orange">
	<img src="https://badgen.net/badge/Python/3.10.0/green">
	<img src="https://badgen.net/badge/MediaPipe/0.9.1.0/blue">
  <img src="https://badgen.net/badge/OpenCV/4.7.0.72/yellow">
  
</p>

# Human Hand Gesture Recognition Software

This is computer vision; real-time hand gesture recognition software implemented in Python and TensoFlow. 
Computer vision is a subfield of artificial intelligence (AI) that enables computers to “see” by deriving meaningful information from digital images, videos and/or act based on their perception.

First, I used MediaPipe's pose and hands ML model to detect and extract fetures of my hands and body pose. On top of that model I built LSTM neural network that learns those fetures (landmarks/keypoints) and later recognises them. You can train then deep LSTM neural network with your own hand gestures, or reuse my pre-trained model stored in .H5 file.

Hand gestures that the .H5 pre-trained model can detect:
- Like 👍
- Ok 👌
- Hello 👋

# Illustration of machine learning pipeline
<p align="center">
  <img src="https://user-images.githubusercontent.com/45321513/226411393-8d56279c-aac7-4978-9260-f75cc853d855.png">
</p>

Author: Armin Smajlagić
