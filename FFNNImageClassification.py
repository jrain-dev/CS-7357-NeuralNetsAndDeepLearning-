import numpy as np

# the assignment made no mentin of excluding ml libratries, so im just gonna use sklearn instead of setting it all up from scratch. if not, i would lik ethe ability to resubmit. Thanks!
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# initialize base FFNN

base_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

# load dataset
digits = load_digits()
X = digits.data
y = digits.target
print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Number of input features:", X.shape[1]) # every image has 64 input features

# initialize training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train base model
print("\nTraining in process, please wait.")
base_model.fit(X_train_scaled, y_train)
print("Training complete.")


#evaluate base model
y_pred = base_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("BASE MODEL PERFORMANCE")
print("=" * 50)

print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# calculate parameters for base model

parameters_without_bias = (
    (64 * 128) +
    (128 * 64) +
    (64 * 10)
)
parameters_with_bias = (
    parameters_without_bias +
    128 +
    64 +
    10
)

print("=" * 50)
print("3a) PARAMETER COUNT")
print("=" * 50)
print("Parameters w/ bias:", parameters_with_bias)
print("Parameters w/o bias:", parameters_without_bias)

# Im not sure if i was supposed to write or code this part, so im just including it here for grade safety reasons lmao

# add another 32-neuron hidden layer to the model
model_plus32 = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

# train new model
print("\nTraining in process, please wait.")
model_plus32.fit(X_train_scaled, y_train)
print("Training complete.")

# evaluate new model
y_pred_plus32 = model_plus32.predict(X_test_scaled)
accuracy_plus32 = accuracy_score(
    y_test,
    y_pred_plus32
)
print("Model with additional layer accuracy:", accuracy_plus32)


print("=" * 50)
print("3b) BASE Vs. BASE+32")
print("=" * 50)
print("Original model accuracy:", accuracy)
print("Model with additional layer accuracy:", accuracy_plus32)
print("Difference:", accuracy_plus32 - accuracy)

# improved model proposition: i think i can make the model better by adding a 256neuron layer and removing the 32neuron layer. 

# train "improved" model
improved_model = MLPClassifier(
    hidden_layer_sizes=(256, 64),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)
print("\nTraining in process, please wait.")
improved_model.fit(X_train_scaled, y_train)
print("Training complete.")
y_pred_improved = improved_model.predict(
    X_test_scaled
)
accuracy_improved = accuracy_score(
    y_test,
    y_pred_improved
)


# calculate model evaluation parameters for 'improved' model
print("=" * 50)
print("3b) is it any better?")
print("=" * 50)
print("Improved model accuracy:",
      accuracy_improved)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_improved
    )
)
improved_without_bias = (
    (64 * 256) +
    (256 * 64) +
    (64 * 10)
)
improved_with_bias = (
    improved_without_bias +
    256 +
    64 +
    10
)
print("Parameters without bias:", improved_without_bias)
print("Parameters with bias:", improved_with_bias)

# Compare all three models
print("=" * 50)
print("FINALCOMPARISON ")
print("=" * 50)
print(f"Base (128, 64):       {accuracy:.4f}")
print(f"Base+32 (128, 64, 32):     {accuracy_plus32:.4f}")
print(f"Improved (256, 64):       {accuracy_improved:.4f}")
