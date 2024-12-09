# Scene Graph Generation with ResNet and Graph Neural network

This project implements a Scene Graph Generation model that combines ResNet for object detection (bounding boxes and object classification) and Graph Convolutional Networks (GCN) for relationship prediction between objects. The goal is to produce a structured representation of images as scene graphs.

## Features

- **Object Detection**: Uses ResNet for detecting and classifying objects in the image, producing bounding boxes.
- **Relationship Prediction**: Leverages GCN to model relationships between detected objects, forming the edges of the scene graph.
- **Visualization**: Graphically represent the scene graph with bounding boxes and relationships.

---

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/hasan200069/GenA
   cd GenAi

   ```
pip install -r requirements.txt

  
