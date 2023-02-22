import cv2
import numpy as np
import os
import random

#path to folder
folder_path = "/Users/oliverkjoeller/Desktop/visual_analytics/cds-visual/data/flowers/"
image_filenames = os.listdir(folder_path)

#Choose a random image from folder
random_image_filename = np.random.choice(image_filenames)
# Load random image
random_image_path = os.path.join(folder_path, random_image_filename)
random_image = cv2.imread(random_image_path)
# Calculate the color histogram of random image
random_hist = cv2.calcHist([random_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
random_hist = cv2.normalize(random_hist, random_hist).flatten()
similarity_scores = []

for image_filename in image_filenames:
    if image_filename != random_image_filename:
        image_path = os.path.join(folder_path, image_filename)
        image = cv2.imread(image_path)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        # Calculate the similarity score between the random image and the current image
        score = cv2.compareHist(random_hist, hist, cv2.HISTCMP_CORREL)
        # Add the similarity score to the list
        similarity_scores.append((image_filename, score))

similarity_scores = sorted(similarity_scores, reverse=True)

print("The 5 most similar images to {} are:".format(random_image_filename))
for i in range(5):
    print("{} ({:.2f})".format(similarity_scores[i][0], similarity_scores[i][1]))