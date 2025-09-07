# Introduction to Artificial Intelligence Projects

This repository contains my coursework for the **NYCU CS Introduction to Artificial Intelligence** class. It collects assignments that cover computer vision, natural language processing and reinforcement learning. The repository serves as a showcase of my AI side projects for future interviews.

## Directory Overview
- **HW0 – Python & Image Processing Basics**
  - Draw bounding boxes on images, perform frame differencing in video and apply common image transformations using PIL and OpenCV.
- **HW1 – Classical Machine Learning for Vision**
  - Detect car occupancy in parking‑lot images with KNN, Random Forest and AdaBoost classifiers. Includes data loading utilities and a simple detection pipeline.
- **HW2 – Natural Language Processing**
  - Sentiment analysis on IMDB reviews using three approaches: an n‑gram language model, an RNN built from scratch and a fine‑tuned DistilBERT model.
- **HW3 – Reinforcement Learning in Pacman**
  - Implementations of value iteration, Q‑learning and multi‑agent search agents for the Pacman environment. Pre‑trained policy and target networks are provided.
- **HW4 – Final Report**
  - Course report summarising experimental results and lessons learned.

## Getting Started
All projects use Python 3. To ensure the scripts are syntactically valid you can run:

```bash
python -m py_compile $(git ls-files '*.py')
```

Each homework folder is self‑contained. Refer to the PDF files inside each folder for detailed problem descriptions, datasets and instructions.

## Acknowledgements
Parts of this repository are adapted from course materials of NYCU and the UC Berkeley Pacman AI projects.
