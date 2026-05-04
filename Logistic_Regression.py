import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/normalised_hand_data_with_clusters.csv")

# # 1. THE MATH: Function to calculate flexion angles
# def get_finger_angle(df, prefix):
#     # Vector A: Palm to Knuckle (Base)
#     v1 = np.array([df[f'{prefix}_KNU1_B_X'] - df['PALM_POSITION_X'],
#                    df[f'{prefix}_KNU1_B_Y'] - df['PALM_POSITION_Y'],
#                    df[f'{prefix}_KNU1_B_Z'] - df['PALM_POSITION_Z']]).T
#     # Vector B: Knuckle to Fingertip
#     v2 = np.array([df[f'{prefix}_KNU3_A_X'] - df[f'{prefix}_KNU1_B_X'],
#                    df[f'{prefix}_KNU3_A_Y'] - df[f'{prefix}_KNU1_B_Y'],
#                    df[f'{prefix}_KNU3_A_Z'] - df[f'{prefix}_KNU1_B_Z']]).T
    
#     # Normalize and calculate Angle (Degrees)
#     v1_u = v1 / np.linalg.norm(v1, axis=1)[:, None]
#     v2_u = v2 / np.linalg.norm(v2, axis=1)[:, None]
#     dot = np.einsum('ij,ij->i', v1_u, v2_u)
#     return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

# # 2. FEATURE ENGINEERING: Calculate angles for all 5 fingers

# FINGER_PREFIXES = {
#     "Thumb":   "TH",
#     "Pinky":   "F1",
#     "Ring":     "F2",
#     "Middle":   "F3",
#     "Index":   "F4",
# }

# FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

# FINGER_COLORS = [
#     "#e63946",  # Thumb
#     "#2a9d8f",  # Index
#     "#e9c46a",  # Middle
#     "#f4a261",  # Ring
#     "#457b9d",  # Pinky
# ]

# angle_col = []
# for finger, prefix in FINGER_PREFIXES.items():
#     col_name = f'{finger}_Angle'
#     df[col_name] = get_finger_angle(df, prefix)
#     angle_col.append(col_name)

# # 3. TRAINING: Use Logistic Regression to find the "Open/Closed" threshold
# finger_models = {}
# for finger in FINGER_PREFIXES.keys():
#     angle_col = f'{finger}_Angle'
    
#     # Create initial "Heuristic" labels for the model to learn from
#     # (Angles < 45 are usually open, but Logistic Regression will find the 'best' spot)
#     # Instead of a hard 45, we use the median. 
#     # This guarantees the model sees a mix of 'high' and 'low' angles.
#     temp_y = (df[angle_col] < df[angle_col].median()).astype(int)
    
#     model = LogisticRegression()
#     model.fit(df[[angle_col]], temp_y)
#     finger_models[finger] = model
    
#     # Output the Statistically Optimal Threshold found by the model
#     boundary = -model.intercept_ / model.coef_
#     print(f"Optimal {finger} Open Threshold: {boundary[0][0]:.2f}°")

# # 4. THE STITCH: Generate the 5-digit Binary Finger Code
# def generate_finger_code(row):
#     code = ""
#     for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
#         angle = row[f'{finger}_Angle']

#         angle_col = f'{finger}_Angle'
#         test_df = pd.DataFrame([[angle]], columns=[angle_col])
        
#         # 2. Predict using that DataFrame
#         bit = finger_models[finger].predict(test_df)[0]
#         code += str(bit)
#     return code

# # Apply to every row in the dataset
# df['Finger_Code'] = df.apply(generate_finger_code, axis=1)

# # 5. CROSS-TABULATION: Map Finger Codes to K-Means Clusters
# # 'Cluster_Number' is your column from the K-Means clustering
# mapping_table = pd.crosstab(df['Cluster_Number'], df['Finger_Code'])
# print("\n--- Mapping Table (Clusters vs. Binary Logic) ---")
# print(mapping_table)



# cluster_dictionaries = {}

# # Iterate over each cluster row in the cross-tabulation table
# for cluster_id in mapping_table.index:
#     # Get all binary codes and frequencies for the current cluster
#     cluster_series = mapping_table.loc[cluster_id]
    
#     # Filter to keep only non-zero occurrences and convert to a dictionary
#     non_zero_codes = cluster_series[cluster_series > 0].to_dict()
    
#     # Sort the dictionary by frequency (highest first) for readability
#     sorted_codes = dict(sorted(non_zero_codes.items(), key=lambda item: item[1], reverse=True))
    
#     # Store in the master dictionary
#     cluster_dictionaries[f"Cluster {cluster_id}"] = sorted_codes

# # --- Print the results cleanly in terminal ---
# print("\n--- Cluster Binary Code Frequencies ---")
# for cluster, codes in cluster_dictionaries.items():
#     print(f"{cluster} = {codes}")


# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix

# # Load the dataset
# df = pd.read_csv("data/normalised_hand_data_with_clusters.csv")

from sklearn.metrics import classification_report, confusion_matrix

#TAKE 2: Kinda cool actually
# # 1. THE MATH: Function to calculate flexion angles
# def get_finger_angle(df, prefix):
#     v1 = np.array([df[f'{prefix}_KNU1_B_X'] - df['PALM_POSITION_X'],
#                    df[f'{prefix}_KNU1_B_Y'] - df['PALM_POSITION_Y'],
#                    df[f'{prefix}_KNU1_B_Z'] - df['PALM_POSITION_Z']]).T
#     v2 = np.array([df[f'{prefix}_KNU3_A_X'] - df[f'{prefix}_KNU1_B_X'],
#                    df[f'{prefix}_KNU3_A_Y'] - df[f'{prefix}_KNU1_B_Y'],
#                    df[f'{prefix}_KNU3_A_Z'] - df[f'{prefix}_KNU1_B_Z']]).T
    
#     v1_u = v1 / np.linalg.norm(v1, axis=1)[:, None]
#     v2_u = v2 / np.linalg.norm(v2, axis=1)[:, None]
#     dot = np.einsum('ij,ij->i', v1_u, v2_u)
#     return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

# # 2. FEATURE ENGINEERING: Calculate the 5 angles
# FINGER_PREFIXES = {
#     "Thumb":   "TH",
#     "Pinky":   "F1",
#     "Ring":    "F2",
#     "Middle":  "F3",
#     "Index":   "F4",
# }

# angle_cols = []
# for finger, prefix in FINGER_PREFIXES.items():
#     col_name = f'{finger}_Angle'
#     df[col_name] = get_finger_angle(df, prefix)
#     angle_cols.append(col_name)

# # Identify the original cluster/target column
# cluster_col = 'Cluster_Label' if 'Cluster_Label' in df.columns else 'Cluster_Number'

# # Define our features (X) and ground truth target (y)
# X = df[angle_cols].values
# y = df[cluster_col].values

# # 3. TRAINING: Fit a single Multinomial Logistic Regression model
# # multi_class='multinomial' trains the model to recognize all 5 clusters at once
# model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
# model.fit(X, y)

# # 4. PREDICTION: Generate predicted cluster labels for every row
# df['Predicted_Cluster'] = model.predict(X)

# # 5. EVALUATION: Compare the K-Means cluster labels to our model's predictions
# print("--- Classification Report ---")
# print(classification_report(y, df['Predicted_Cluster']))

# print("\n--- Confusion Matrix ---")
# # Creates a comparison matrix of True Clusters vs. Predicted Clusters
# conf_matrix = confusion_matrix(y, df['Predicted_Cluster'])
# print(pd.DataFrame(conf_matrix, 
#                    index=[f"True Clus {i}" for i in range(5)], 
#                    columns=[f"Pred Clus {i}" for i in range(5)]))

# # 6. Save the results
# output_path = "data/hand_data_with_predicted_clusters.csv"
# df.to_csv(output_path, index=False)
# print(f"\nSuccessfully saved predictions to: {output_path}")


### TAKE 3: Incomplete, tried copying the lab but gave up
# from sklearn.model_selection import train_test_split
# Xtrain, Xtest, ytrain, ytest = train_test_split(df.iloc[:,:-1],
#                                                 df["Cluster_Number"],
#                                                 test_size=0.2,
#                                                 random_state=rng_seed )

# from sklearn.preprocessing import StandardScaler
# # Create a scaler object
# scaler = StandardScaler()

# # Use the fit_transform method to perform the normalization of columns
# X = scaler.fit_transform(Xtrain)

# # Format the normalized input as a DataFram
# Xtrain_norm = pd.DataFrame(X, index=Xtrain.index, columns=Xtrain.columns) 

# data_corr_sorted = df.corr().abs().sort_values(by = 'target', ascending = False)
# best_single_feature = data_corr_sorted.index[1]

# from sklearn.linear_model import LogisticRegression
# model_norm = LogisticRegression(random_state=rng_seed)
# model_norm.fit(Xtrain_norm[[best_single_feature]], ytrain)
# print(model_norm.intercept_[0], model_norm.coef_[0,:])

#### TAKE 4: sorta cool
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

# # 1. Load the dataset
# df = pd.read_csv("data/normalised_hand_data_with_clusters.csv")

# # 2. THE MATH: Function to calculate flexion angles
# def get_finger_angle(df, prefix):
#     v1 = np.array([df[f'{prefix}_KNU1_B_X'] - df['PALM_POSITION_X'],
#                    df[f'{prefix}_KNU1_B_Y'] - df['PALM_POSITION_Y'],
#                    df[f'{prefix}_KNU1_B_Z'] - df['PALM_POSITION_Z']]).T
#     v2 = np.array([df[f'{prefix}_KNU3_A_X'] - df[f'{prefix}_KNU1_B_X'],
#                    df[f'{prefix}_KNU3_A_Y'] - df[f'{prefix}_KNU1_B_Y'],
#                    df[f'{prefix}_KNU3_A_Z'] - df[f'{prefix}_KNU1_B_Z']]).T
    
#     v1_u = v1 / np.linalg.norm(v1, axis=1)[:, None]
#     v2_u = v2 / np.linalg.norm(v2, axis=1)[:, None]
#     dot = np.einsum('ij,ij->i', v1_u, v2_u)
#     return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

# # 3. FEATURE ENGINEERING: Calculate the 5 angles
# FINGER_PREFIXES = {
#     "Thumb":   "TH",
#     "Pinky":   "F1",
#     "Ring":    "F2",
#     "Middle":  "F3",
#     "Index":   "F4",
# }

# angle_cols = []
# for finger, prefix in FINGER_PREFIXES.items():
#     col_name = f'{finger}_Angle'
#     df[col_name] = get_finger_angle(df, prefix)
#     angle_cols.append(col_name)

# # Identify the original K-Means cluster/target column
# cluster_col = 'Cluster_Label' if 'Cluster_Label' in df.columns else 'Cluster_Number'

# # Define our features (X) and ground truth target (y)
# X = df[angle_cols].values
# y = df[cluster_col].values

# # 4. DATA SPLITTING: 80% for training, 20% for testing
# # random_state=42 guarantees reproducibility across runs
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=42, stratify=y
# )

# # Note: Using stratify=y ensures that both the training set and testing set 
# # contain the same proportion of each cluster as the original dataset.

# print(f"Training data size: {X_train.shape[0]} samples")
# print(f"Testing data size: {X_test.shape[0]} samples")

# # 5. TRAINING: Fit the model using ONLY the training data
# model = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=454)
# model.fit(X_train, y_train)

# # 6. PREDICTION: Generate predicted cluster labels for the testing data
# y_pred = model.predict(X_test)

# # 7. EVALUATION: Compare the K-Means cluster labels to predictions on test data
# print("\n" + "="*50)
# print("--- TEST DATA: Classification Report ---")
# print("="*50)
# print(classification_report(y_test, y_pred))

# print("\n" + "="*50)
# print("--- TEST DATA: Confusion Matrix ---")
# print("="*50)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Display it cleanly as a pandas DataFrame
# test_matrix_df = pd.DataFrame(
#     conf_matrix, 
#     index=[f"True Clus {i}" for i in range(5)], 
#     columns=[f"Pred Clus {i}" for i in range(5)]
# )
# print(test_matrix_df)


#####TAKE 5: trying to find the best model, lasso logreg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False


# 1. Load the dataset
df = pd.read_csv("data/normalised_hand_data_with_clusters.csv")

# 2. THE MATH: Function to calculate flexion angles
def get_finger_angle(df, prefix):
    v1 = np.array([df[f'{prefix}_KNU1_B_X'] - df['PALM_POSITION_X'],
                   df[f'{prefix}_KNU1_B_Y'] - df['PALM_POSITION_Y'],
                   df[f'{prefix}_KNU1_B_Z'] - df['PALM_POSITION_Z']]).T
    v2 = np.array([df[f'{prefix}_KNU3_A_X'] - df[f'{prefix}_KNU1_B_X'],
                   df[f'{prefix}_KNU3_A_Y'] - df[f'{prefix}_KNU1_B_Y'],
                   df[f'{prefix}_KNU3_A_Z'] - df[f'{prefix}_KNU1_B_Z']]).T
    
    v1_u = v1 / np.linalg.norm(v1, axis=1)[:, None]
    v2_u = v2 / np.linalg.norm(v2, axis=1)[:, None]
    dot = np.einsum('ij,ij->i', v1_u, v2_u)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

# 3. FEATURE ENGINEERING: Calculate the 5 angles
FINGER_PREFIXES = {
    "Thumb":   "TH",
    "Pinky":   "F1",
    "Ring":    "F2",
    "Middle":  "F3",
    "Index":   "F4",
}

angle_cols = []
for finger, prefix in FINGER_PREFIXES.items():
    col_name = f'{finger}_Angle'
    df[col_name] = get_finger_angle(df, prefix)
    angle_cols.append(col_name)

# Identify the original K-Means cluster/target column
cluster_col = 'Cluster_Label' if 'Cluster_Label' in df.columns else 'Cluster_Number'

# Define features (X) and target (y)
X = df[angle_cols].values
y = df[cluster_col].values

# 4. DATA SPLITTING: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. HYPERPARAMETER SEARCH: Loop over different values of C
# Cs from 0.01 to 100 on a logarithmic scale
Cs = np.logspace(-2, 2, 20)
train_scores = []
models = []

print("Searching for the best regularization value 'C'...")
for C in Cs:
    # Use standard scaling to make sure model optimization is stable
    # Use multinomial lbfgs solver (default in newer versions of scikit-learn)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42))
    ])
    
    # 4-Fold cross validation on the training data
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=4, scoring='accuracy')
    
    # Save the mean score and the fitted pipeline
    train_scores.append(cv_scores.mean())
    
    # Fit on all the training data and save
    pipeline.fit(X_train, y_train)
    models.append(pipeline)

# 6. IDENTIFY THE BEST MODEL
best_index = np.argmax(train_scores)
best_C = Cs[best_index]
best_model = models[best_index]

print(f"\nOptimal C: {best_C:.4f} with Validation Accuracy: {train_scores[best_index]:.4f}")

# 7. FINAL EVALUATION: Test the best model on the unseen 20% test set
y_pred = best_model.predict(X_test)

print("\n" + "="*50)
print("--- BEST MODEL TEST PERFORMANCE ---")
print("="*50)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
conf_matrix = confusion_matrix(y_test, y_pred)
test_matrix_df = pd.DataFrame(
    conf_matrix, 
    index=[f"True Clus {i}" for i in range(5)], 
    columns=[f"Pred Clus {i}" for i in range(5)]
)
print(test_matrix_df)

# Optional: Plotting the hyperparameter curve just like your notebook
plt.figure(figsize=(8, 4))
plt.semilogx(Cs, train_scores, 'o-', color='b', linewidth=2)
plt.semilogx(best_C, train_scores[best_index], '*', color='m', markersize=14, label=f'Best C ({best_C:.2f})')
plt.title("Performance vs Regularization Parameter C")
plt.xlabel("Regularization Parameter (C)")
plt.ylabel("Validation Accuracy")
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()