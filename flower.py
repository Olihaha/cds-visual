import cv2
import os
import numpy as np

def find_similar_images(input_filename, folder_path):
    # Load the input image
    input_image_path = os.path.join(folder_path, input_filename)
    input_image = cv2.imread(input_image_path)

    # Calculate the color histogram of the input image
    input_hist = cv2.calcHist([input_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    input_hist = cv2.normalize(input_hist, input_hist).flatten()

    # Load the list of image filenames in the folder
    image_filenames = os.listdir(folder_path)

    # Initialize a list to store the similarity scores
    similarity_scores = []

    # Compare the color histograms of each image in the folder with the input image
    for image_filename in image_filenames:
        if image_filename != input_filename:
            # Load the image
            image_path = os.path.join(folder_path, image_filename)
            image = cv2.imread(image_path)

            # Calculate the color histogram of the image
            hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Calculate the similarity score between the input image and the current image
            score = cv2.compareHist(input_hist, hist, cv2.HISTCMP_CORREL)

            # Add the similarity score to the list
            similarity_scores.append((image_filename, score))

    # Sort the similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the filenames of the 5 most similar images
    top_5_filenames = [x[0] for x in similarity_scores[:5]]

    # Return the filenames of the top 5 most similar images
    return top_5_filenames

folder_path = "/Users/oliverkjoeller/Desktop/visual_analytics/cds-visual/data/flowers/"
input_filename = "/Users/oliverkjoeller/Desktop/visual_analytics/cds-visual/data/flowers/image_0004.jpg"

top_5_filenames = find_similar_images(input_filename, folder_path)

print("The 5 most similar images to {} are:".format(input_filename))
for filename in top_5_filenames:
    print(filename)