# ==============================================================================
# COMPREHENSIVE MACHINE LEARNING LAB CODEBASE
# ==============================================================================
# Instructions: You can run this entire script, but because it contains multiple 
# visualizations (plt.show()), the script will pause at each graph. Close the 
# graph window to let the script continue to the next experiment.
# ==============================================================================


# ==============================================================================
# LAB 1: DATA PREPROCESSING & SIMPLE LINEAR REGRESSION
# ==============================================================================

# --- 1.1 MinMax Scaler ---
from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing

# dataframe = read_csv("pima-indians-diabetes.csv") 
# array = dataframe.values
# data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
# data_rescaled = data_scaler.fit_transform(array)
# set_printoptions(precision=2)
# print("\nScaled data:\n", data_rescaled[0:10])

# --- 1.2 L1 Normalizer ---
from sklearn.preprocessing import Normalizer
# data_normalizer_l1 = Normalizer(norm='l1').fit(array)
# data_normalized_l1 = data_normalizer_l1.fit_transform(array)
# print("\nNormalized data (L1):\n", data_normalized_l1[0:3])

# --- 1.3 L2 Normalizer ---
# data_normalizer_l2 = Normalizer(norm='l2').fit(array)
# data_normalized_l2 = data_normalizer_l2.fit_transform(array)
# print("\nNormalized data (L2):\n", data_normalized_l2[0:3])

# --- 1.4 Binarizer ---
from sklearn.preprocessing import Binarizer
# binarizer = Binarizer(threshold=0.5).fit(array)
# data_binarized = binarizer.fit_transform(array)
# print("\nBinary data:\n", data_binarized[0:3])

# --- 1.5 Standard Scaler ---
from sklearn.preprocessing import StandardScaler
# data_scaler_std = StandardScaler().fit(array)
# data_rescaled_std = data_scaler_std.transform(array)
# print("\nRescaled data:\n", data_rescaled_std[0:4])

# --- 1.6 Simple Linear Regression (Manual) ---
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# df_sal = pd.read_csv("Salary_Data.csv")
# x_actual = df_sal["YearsExperience"]
# y_actual = df_sal["Salary"]
# x_bar = np.mean(x_actual)
# y_bar = np.mean(y_actual)
# n = len(x_actual)

# N = 0
# D = 0
# for i in range(n):
#     N = N + (x_actual[i] - x_bar) * (y_actual[i] - y_bar)
#     D = D + (x_actual[i] - x_bar)**2

# beta1 = N / D
# beta0 = y_bar - beta1 * x_bar
# print("\nLine of regression is y=", round(beta0,2), "+", round(beta1,2), "x")

# y_pred = np.zeros(n)
# for i in range(n):
#     y_pred[i] = beta0 + beta1 * x_actual[i]

# plt.scatter(x_actual, y_actual, color='lightcoral', label='Actual Data')
# plt.plot(df_sal['YearsExperience'], y_pred, color='firebrick', label='Regression Line data')
# plt.title('Salary vs Experience (Regression Line)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.legend(loc='best', facecolor='white')
# plt.show()


# ==============================================================================
# LAB 2: EXPLORATORY DATA ANALYSIS (FRAUD DETECTION)
# ==============================================================================

# LABELS = ["normal", "fraud"]
# data = pd.read_csv('creditcard.csv')

# # --- 2.1 Bar Chart ---
# count_classes = pd.value_counts(data['Class'], sort=True)
# count_classes.plot(kind='bar', rot=0)
# plt.title("Transaction class Distribution")
# plt.xticks(range(2), LABELS)
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# plt.show()

# # --- 2.2 Histogram ---
# fraud = data[data['Class'] == 1]
# normal = data[data['Class'] == 0]

# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# f.suptitle('Amount per transaction by Class')
# bins = 50
# ax1.hist(fraud.Amount, bins=bins)
# ax1.set_title('Fraud')
# ax2.hist(normal.Amount, bins=bins)
# ax2.set_title('Normal')
# plt.xlabel('Amount($)')
# plt.ylabel('Number of Transactions')
# plt.xlim((0, 20000))
# plt.yscale('log')
# plt.show()

# # --- 2.3 Scatter Plot ---
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# f.suptitle('Time of transaction vs Amount by Class')
# ax1.scatter(fraud.Time, fraud.Amount)
# ax1.set_title('Fraud')
# ax2.scatter(normal.Time, normal.Amount)
# ax2.set_title('Normal')
# plt.xlabel('Time(in seconds)')
# plt.ylabel('Amount')
# plt.show()

# # --- 2.4 Print Outlier Fraction ---
# data1 = data.sample(frac=0.1, random_state=1)
# Fraud = data1[data1['Class'] == 1]
# Valid = data1[data1['Class'] == 0]
# outlier_fraction = len(Fraud) / float(len(Valid))
# print("\nOutlier Fraction:", outlier_fraction)


# ==============================================================================
# LAB 3: REGRESSION MODELS (SCIKIT-LEARN)
# ==============================================================================

# --- 3.1 Linear Regression (Synthetic Data) ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x_syn = 2 * np.random.rand(100, 1)
y_syn = 4 + 3 * x_syn + np.random.randn(100, 1)

x_train_syn, x_test_syn, y_train_syn, y_test_syn = train_test_split(x_syn, y_syn, test_size=0.2, random_state=42)

model_lin = LinearRegression()
model_lin.fit(x_train_syn, y_train_syn)
y_pred_syn = model_lin.predict(x_test_syn)

plt.scatter(x_train_syn, y_train_syn, color='blue', label='training data')
plt.scatter(x_test_syn, y_test_syn, color='red', label='testing data')
plt.legend()
plt.title('Synthetic Linear Regression')
plt.show()

# --- 3.2 Logistic Regression (Loan Data) ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# df_loan = pd.read_csv('loan.csv')
# a = df_loan[['age']]
# b = df_loan['loan']

# x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(a, b, test_size=0.2)

# model_log = LogisticRegression()
# model_log.fit(x_train_log, y_train_log)

# print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_test_log, model_log.predict(x_test_log)))
# print("Classification Report:\n", classification_report(y_test_log, model_log.predict(x_test_log)))

# plt.scatter(df_loan.age, df_loan.loan, color='red')
# plt.title('Logistic Regression: Age vs Loan')
# plt.show()


# ==============================================================================
# LAB ASSESSMENT 2
# ==============================================================================

# --- Experiment 1: Back Propagation ---
def sigmoid(z): 
    return(1/(1+np.exp(-z))) 

def sigmoid_derivative(a): 
    return(a*(1-a)) 

# XOR Data
X_bp = np.array([[0,0],[0,1],[1,0],[1,1]]) 
y_bp = np.array([[0],[1],[1],[0]]) 
input_neurons, hidden_neurons, output_neurons = 2, 3, 1
learning_rate = 0.5 
epochs = 2000 

np.random.seed(1) 
W1 = np.random.randn(input_neurons, hidden_neurons) 
b1 = np.random.randn(1, hidden_neurons) 
W2 = np.random.randn(hidden_neurons, output_neurons) 
b2 = np.random.randn(1, output_neurons) 

for epoch in range(epochs): 
    z1 = np.dot(X_bp, W1) + b1 
    a1 = sigmoid(z1) 
    z2 = np.dot(a1, W2) + b2 
    y_hat = sigmoid(z2) 

    error_output = y_hat - y_bp 
    delta_output = error_output * sigmoid_derivative(y_hat) 
    error_hidden = delta_output.dot(W2.T) 
    delta_hidden = error_hidden * sigmoid_derivative(a1) 

    W2 -= learning_rate * a1.T.dot(delta_output) 
    b2 -= learning_rate * np.sum(delta_output, axis=0, keepdims=True) 
    W1 -= learning_rate * X_bp.T.dot(delta_hidden) 
    b1 -= learning_rate * np.sum(delta_hidden, axis=0, keepdims=True) 

print("\nBackprop Final Predictions (XOR):") 
print(y_hat.round(3)) 

# --- Experiment 2: K-Nearest Neighbors (KNN) ---
from sklearn.neighbors import KNeighborsClassifier

X_train_knn = np.array([[1, 2], [2, -1], [2, 6], [1, 1], [3, 4], [-2, -4], [-4,-2], [-2,-1], [-1,-1], [-2,2], [-4,-3]])
y_train_knn = np.array([0,0,0,0,0,1,1,1,1,1,1])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_knn, y_train_knn)

X_test_knn = np.array([[0,2]])
prediction_knn = knn.predict(X_test_knn)

print("\nKNN Prediction for [0,2]:", "A" if prediction_knn[0] == 0 else "B")

# --- Experiment 3: Support Vector Classifier (SVC) ---
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# letter = pd.read_csv("letterdata.csv")
# X_svc = letter.drop('letter', axis=1)
# Y_svc = letter['letter']

# X_train_svc, X_test_svc, Y_train_svc, Y_test_svc = train_test_split(X_svc, Y_svc, test_size=0.20)
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train_svc, Y_train_svc)
# Y_pred_svc = svclassifier.predict(X_test_svc)

# print("\nSVC Accuracy:", accuracy_score(Y_test_svc, Y_pred_svc) * 100)

# --- Experiment 4: Naive Bayes ---
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

# df_nb = pd.read_csv('loan_data.csv')
# sns.countplot(data=df_nb, x='purpose', hue='not.fully.paid')
# plt.xticks(rotation=45, ha='right')
# plt.title("Naive Bayes: Loan Data Purpose")
# plt.show()

# pre_df = pd.get_dummies(df_nb, columns=['purpose'], drop_first=True)
# X_nb = pre_df.drop('not.fully.paid', axis=1)
# y_nb = pre_df['not.fully.paid']

# X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.33, random_state=125)

# model_nb = GaussianNB()
# model_nb.fit(X_train_nb, y_train_nb)
# y_pred_nb = model_nb.predict(X_test_nb)
# print("\nNaive Bayes Accuracy:", accuracy_score(y_pred_nb, y_test_nb))

# --- Experiment 5: Decision Tree ---
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, tree

iris = datasets.load_iris()
X_dt = iris.data
y_dt = iris.target

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.3, random_state=1)

clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_dt.fit(X_train_dt, y_train_dt)
y_pred_dt = clf_dt.predict(X_test_dt)

print("\nDecision Tree Accuracy:", accuracy_score(y_test_dt, y_pred_dt))

fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(clf_dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()


# ==============================================================================
# LAB ASSESSMENT 3
# ==============================================================================

# --- Experiment 1: AdaBoost ---
from sklearn.ensemble import AdaBoostClassifier

X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(iris.data, iris.target, test_size=0.3)

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
abc.fit(X_train_ab, y_train_ab)
y_pred_ab = abc.predict(X_test_ab)

print("\nAdaBoost Accuracy:", accuracy_score(y_test_ab, y_pred_ab))

# --- Experiment 2: XGBoost ---
from xgboost import XGBClassifier

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

xgb_model = XGBClassifier(n_estimators=50, learning_rate=1, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_xgb, y_train_xgb)
y_pred_xgb = xgb_model.predict(X_test_xgb)

print("\nXGBoost Accuracy:", accuracy_score(y_test_xgb, y_pred_xgb))


# ==============================================================================
# LAB ASSESSMENT 4
# ==============================================================================

# --- Experiment 1: K-Means Clustering ---
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X_km, y_km = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X_km)

plt.scatter(X_km[y_kmeans == 0, 0], X_km[y_kmeans == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X_km[y_kmeans == 1, 0], X_km[y_kmeans == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X_km[y_kmeans == 2, 0], X_km[y_kmeans == 2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X_km[y_kmeans == 3, 0], X_km[y_kmeans == 3, 1], s=50, c='cyan', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

# --- Experiment 2: Hierarchical Clustering ---
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

X_hc = np.array([[5,3], [10,15], [15,12], [24,10], [30,30], [85,70], [71,80], [60,78], [70,55], [80,91]])
linked = linkage(X_hc, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

cluster_hc = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
print("\nHierarchical Cluster Assignments:\n", cluster_hc.fit_predict(X_hc))

# --- Experiment 3: K-Modes Clustering ---
from kmodes.kmodes import KModes

data_kmode = np.array([
    ['red', 'large', 'round'], ['blue', 'small', 'square'], ['red', 'small', 'round'],
    ['blue', 'large', 'square'], ['red', 'medium', 'round'], ['green', 'small', 'triangle'],
    ['green', 'medium', 'triangle']
])

km_modes = KModes(n_clusters=3, init='Huang', n_init=5, verbose=0)
clusters_modes = km_modes.fit_predict(data_kmode)

print("\nFinal K-Modes Centroids:\n", km_modes.cluster_centroids_)


# ==============================================================================
# LAB ASSESSMENT 5
# ==============================================================================

# --- Experiment 1: Evaluating ML Algorithms (SMOTE) ---
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

X_sm, y_sm = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.90, 0.10], random_state=42)
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.3, random_state=42)

print("\n--- UNBALANCED DATA EVALUATION ---")
rf_unbalanced = RandomForestClassifier(random_state=42)
rf_unbalanced.fit(X_train_sm, y_train_sm)
y_pred_unb = rf_unbalanced.predict(X_test_sm)
print(confusion_matrix(y_test_sm, y_pred_unb))

print("\n--- BALANCED DATA EVALUATION (Using SMOTE) ---")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_sm, y_train_sm)

rf_balanced = RandomForestClassifier(random_state=42)
rf_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_bal = rf_balanced.predict(X_test_sm)
print(confusion_matrix(y_test_sm, y_pred_bal))

# --- Experiment 2: Comparison of ML Algorithms ---
from sklearn.datasets import load_breast_cancer

data_bc = load_breast_cancer()
X_bc = data_bc.data
y_bc = data_bc.target
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("\n--- MODEL COMPARISON RESULTS ---")
for name, model in models.items():
    model.fit(X_train_bc, y_train_bc)
    y_pred_bc = model.predict(X_test_bc)
    print(f"{name:25s} : Accuracy = {accuracy_score(y_test_bc, y_pred_bc) * 100:.2f}%")
