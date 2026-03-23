from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

num_of_samples_in_class = y.value_counts()
print("\nNumber of samples in each class.")
for labels, count in num_of_samples_in_class.items():
    print(f"{data.target_names[labels]}: {count}")
#print(data.DESCR)

#The dataset is imbalanced, but 357 and 212 is not too far apart.
#There are more benign samples than malignant samples.

#class balance is important in classification because if one class has more samples than the other
#the model may just learn to predict the class with more samples because it is technically
#more likely. This may result in a high accuracy score on test data, but when it comes
#to real application the mpdel is insufficient in accurately predicting cancer.