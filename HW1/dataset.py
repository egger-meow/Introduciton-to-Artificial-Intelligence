import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    dataset = []
    width = 36
    height = 16
    
    # Begin your code (Part 1)
    for file in os.listdir(data_path+"/car"):
        path = os.path.join(data_path+"/car", file)
        img = cv2.imread(path)

        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dataset.append((img,1))
        
    for file in os.listdir(data_path+"/non-car"):
        path = os.path.join(data_path+"/non-car", file)
        img = cv2.imread(path)
        
        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dataset.append((img,0))
        
    # End your code (Part 1)
    return dataset
