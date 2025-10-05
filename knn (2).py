
#-------------------------------------------------------------------------
# AUTHOR: Aaron Manoj
# FILENAME: knn.py
# SPECIFICATION: 1-NN LOO-CV on email_classification.csv
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

# IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

# Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

# Loop your data to allow each instance to be your test set
errors = 0
n = len(db)

# Assume last column is the class label (spam/ham as text)
cols = list(df.columns)
feature_cols = cols[:-1]
label_col = cols[-1]

X_full = df[feature_cols].astype(float).to_numpy()
y_full = df[label_col].to_numpy()

for i in range(n):
    # Training data: remove the i-th instance
    X_train = np.delete(X_full, i, axis=0)
    y_train = np.delete(y_full, i, axis=0)

    # Test sample (1 x 20)
    testSample = X_full[i].reshape(1, -1)
    true_label = y_full[i]

    # Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X_train, y_train)

    # Predict for the held-out sample
    class_predicted = clf.predict(testSample)[0]

    # Count error
    if class_predicted != true_label:
        errors += 1

# Print the error rate
error_rate = errors / n if n else 0.0
print(f"1NN LOO-CV error rate: {error_rate:.4f} ({errors} / {n})")
