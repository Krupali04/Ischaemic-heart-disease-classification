import argparse
from data_preprocessing import load_and_preprocess_data
from models import (get_naive_bayes_pipeline, get_neural_network_pipeline,
                    get_decision_tree_pipeline, get_knn_pipeline, get_cost_sensitive_pipeline)
from evaluation import evaluate_model, plot_feature_importance

def main(args):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(args.data_dir)
    
    # Define feature names
    feature_names = ['Mean', 'Std Dev', 'Max', 'Min', 'RMS', 'Peak-to-Peak']
    
    # Define pipelines
    pipelines = {
        'Naive Bayes': get_naive_bayes_pipeline(),
        'Neural Network': get_neural_network_pipeline(),
        'Decision Tree': get_decision_tree_pipeline(),
        'KNN': get_knn_pipeline()
    }
    
    # Add cost-sensitive variants if requested
    if args.use_cost_sensitive:
        # Cost matrix: higher cost for false negatives (missing ischemia)
        cost_matrix = [[0, 1], [5, 0]]
        for name, pipeline in list(pipelines.items()):
            cs_name = f'Cost-Sensitive {name}'
            pipelines[cs_name] = get_cost_sensitive_pipeline(pipeline, cost_matrix)
    
    # Evaluate each pipeline
    best_model = None
    best_f1 = 0
    
    for name, pipeline in pipelines.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test, name)
        
        # Track best model based on F1 score
        if metrics['F1 Score'] > best_f1:
            best_f1 = metrics['F1 Score']
            best_model = pipeline
    
    # Plot feature importance for the best model
    if best_model is not None and hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        print("\nPlotting feature importance for the best model...")
        plot_feature_importance(best_model.named_steps['classifier'], feature_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate IHD detection models')
    parser.add_argument('--data_dir', type=str, default='../data/ischemia_dataset',
                      help='Directory containing .mat and .hea files')
    parser.add_argument('--use_cost_sensitive', action='store_true',
                      help='Add cost-sensitive variants of the models')
    
    args = parser.parse_args()
    main(args)