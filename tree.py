from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Training data: Age, Income, Credit Score
X = [
    [22, 1500, 0],
    [25, 2000, 0],
    [35, 5000, 1],
    [45, 6000, 1],
    [28, 4000, 1],
    [50, 2500, 0],
    [40, 4500, 1],
    [23, 3500, 0]
]

# 0 = No, 1 = Yes
y = [0, 0, 1, 1, 1, 0, 1, 0]

# Create and train the tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Predict
person = [[30, 4200, 1]]
prediction = model.predict(person)
print("Loan decision:", "Yes" if prediction[0] == 1 else "No")

# Draw the tree
plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=["Age", "Income", "Credit Score"],
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree - Loan Approval")
plt.tight_layout()

# Save and display
plt.savefig("decision_tree.png", dpi=150)
plt.show()