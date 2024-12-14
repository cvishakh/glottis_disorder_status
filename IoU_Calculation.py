import numpy as np
import flammkuchen as fl
import matplotlib.pyplot as plt

# Load rectangles data from the provided HDF5 file
data = fl.load("rectangles_dsss.sec")

# Extract the ground truth and predicted rectangles
ground_truth = np.array(data['ground_truth'])  # Assuming these are the correct keys
predicted = np.array(data['predicted'])

#Define IoU score function
def calculate_iou(rect1, rect2):
    #Extract coordinates and sizes
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    #Compute the (x, y)-coordinates of the intersection rectangle
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)

    #Compute the area of intersection
    inter_width = max(0, x_inter2 - x_inter1)
    inter_height = max(0, y_inter2 - y_inter1)
    inter_area = inter_width * inter_height

    #Compute the area of each rectangle
    area_rect1 = w1 * h1
    area_rect2 = w2 * h2

    #Find area of union
    union_area = area_rect1 + area_rect2 - inter_area

    #Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0  #Added check for case when area of union is zero
    return iou

# Calculate IoU scores for all rectangle pairs
iou_scores = [calculate_iou(ground_truth[i], predicted[i]) for i in range(len(ground_truth))]

#Plot the IoU scores Distribution in a histogram
plt.hist(iou_scores, bins=20, color='blue', alpha=0.7)
plt.title("IoU Score Distribution")
plt.xlabel("IoU")
plt.ylabel("No. of rectangles")
plt.show()