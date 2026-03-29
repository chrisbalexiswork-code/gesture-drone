# Gesture-Controlled Air Drawing

Draw in the air using only your hands. Built with Python, MediaPipe, and OpenCV.

## Demo
[demo.mov](demo.mov)

## How It Works
- **Right hand, index finger up** → draws on screen
- **Left hand, open palm** → clears the canvas
- Uses MediaPipe's hand landmark model to detect 21 points per hand in real time

## Tech Stack
- Python 3.9
- MediaPipe 0.10.9
- OpenCV 4.13

## Run It Yourself
```bash
pip install opencv-python mediapipe==0.10.9 numpy
python3 main.py
```

## What's Next
Extending this to control a drone in simulation using AirSim — mapping hand gestures to flight commands as part of a larger human-robot interaction project.

## Author
Chris Bryan Alexis — Physics @ Montclair State University
