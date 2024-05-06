import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # Define TP, FN, FP, TN
TP = 208
FN = 28
FP = 110
TN = 381

# Define TP, FN, FP, TN for the negative class
# TP = 427
# FN = 64
# FP = 64
# TN = 427


# Calculate total count of actual positives (AP) and actual negatives (AN)
AP = TP + FN
AN = TN + FP

# Calculate rates
TPR = round(TP / AP,3)  # True Positive Rate (Sensitivity)
FPR = round(FP / AN,3) # False Positive Rate
FNR = round(FN / AP,3)  # False Negative Rate (Miss Rate)
TNR = round(TN / AN,3)  # True Negative Rate (Specificity)

# Display rates
print("True Positive Rate (Sensitivity):", TPR)
print("False Positive Rate:", FPR)
print("False Negative Rate (Miss Rate):", FNR)
print("True Negative Rate (Specificity):", TNR)

# Create confusion matrix
conf_matrix = np.array([[TNR, FPR], [FNR, TPR]])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
