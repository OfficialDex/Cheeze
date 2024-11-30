import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset with error names/details and corresponding solutions
error_solution_pairs = [
    {"error": "NameError: name 'x' is not defined", "solution": "Define the variable 'x' before using it."},
    {"error": "IndexError: list index out of range", "solution": "Ensure the index is within the bounds of the list."},
    {"error": "TypeError: unsupported operand type(s) for +: 'int' and 'str'", "solution": "Convert the types to compatible ones before adding."},
    {"error": "KeyError: 'key'", "solution": "Check if the key exists in the dictionary before accessing."},
    {"error": "ValueError: too many values to unpack (expected 2)", "solution": "Ensure the number of elements being unpacked matches the expected number."}
]

# Prepare the dataset
errors = [pair["error"] for pair in error_solution_pairs]
solutions = [pair["solution"] for pair in error_solution_pairs]

# Vectorize the errors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(errors)
y = solutions

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy on test data:", accuracy_score(y_test, y_pred))

# Calculate total parameters
total_params = 0
for layer_weights in model.coefs_:
    total_params += layer_weights.size
for layer_biases in model.intercepts_:
    total_params += layer_biases.size

print(f"Total Parameters in the Model: {total_params}")

# Save the model and vectorizer
with open('pytrix_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('pytrix_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully."
