from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Define pipelines for each model
def get_naive_bayes_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])

def get_neural_network_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42))
    ])

def get_decision_tree_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(criterion='entropy', random_state=42))
    ])

def get_knn_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])

# Cost-sensitive wrapper (simplified for demonstration)
class CostSensitiveClassifier:
    def __init__(self, base_classifier, cost_matrix):
        self.base_classifier = base_classifier
        self.cost_matrix = cost_matrix  # [[0, cost_0->1], [cost_1->0, 0]]
    
    def fit(self, X, y):
        self.base_classifier.fit(X, y)
        return self
    
    def predict(self, X):
        probs = self.base_classifier.predict_proba(X)
        # Adjust predictions based on costs
        adjusted_preds = []
        for prob in probs:
            expected_cost_0 = prob[0] * self.cost_matrix[1][0]  # Cost of misclassifying 1 as 0
            expected_cost_1 = prob[1] * self.cost_matrix[0][1]  # Cost of misclassifying 0 as 1
            adjusted_preds.append(0 if expected_cost_0 < expected_cost_1 else 1)
        return adjusted_preds
    
    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)

def get_cost_sensitive_pipeline(base_pipeline, cost_matrix):
    base_classifier = base_pipeline.named_steps['classifier']
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', CostSensitiveClassifier(base_classifier, cost_matrix))
    ])