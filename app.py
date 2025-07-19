#!/usr/bin/env python3
"""
PyInterpret Web Demo - Interactive ML Model Interpretation
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# Import PyInterpret
from pyinterpret import (
    SHAPExplainer, 
    LIMEExplainer, 
    PermutationImportanceExplainer,
    PartialDependenceExplainer
)

app = Flask(__name__)

# Global variables to store current model and data
current_model = None
current_data = None
current_task = None
feature_names = None

def create_sample_classification_data():
    """Create sample classification dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), y, feature_names

def create_sample_regression_data():
    """Create sample regression dataset"""
    X, y = make_regression(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        noise=0.1,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), y, feature_names

def train_model(task_type, model_type):
    """Train a model based on task and model type"""
    global current_model, current_data, current_task, feature_names
    
    if task_type == 'classification':
        X, y, feature_names = create_sample_classification_data()
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # logistic_regression
            model = LogisticRegression(random_state=42, max_iter=1000)
    else:  # regression
        X, y, feature_names = create_sample_regression_data()
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # linear_regression
            model = LinearRegression()
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate performance
    if task_type == 'classification':
        score = accuracy_score(y_test, model.predict(X_test))
        metric = 'Accuracy'
    else:
        score = r2_score(y_test, model.predict(X_test))
        metric = 'R² Score'
    
    current_model = model
    current_data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    current_task = task_type
    
    return {
        'success': True,
        'model_type': model_type,
        'task_type': task_type,
        'performance': f'{metric}: {score:.3f}',
        'data_shape': f'{X.shape[0]} samples, {X.shape[1]} features',
        'feature_names': feature_names
    }

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_str

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    """Train a new model"""
    data = request.json
    task_type = data.get('task_type', 'classification')
    model_type = data.get('model_type', 'random_forest')
    
    try:
        result = train_model(task_type, model_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/explain_local', methods=['POST'])
def api_explain_local():
    """Generate local explanations"""
    if current_model is None:
        return jsonify({'success': False, 'error': 'No model trained'})
    
    data = request.json
    explainer_type = data.get('explainer_type', 'shap')
    instance_idx = data.get('instance_idx', 0)
    
    try:
        X_test = current_data['X_test']
        instance = X_test.iloc[instance_idx]
        
        if explainer_type == 'shap':
            if current_task == 'classification':
                explainer = SHAPExplainer(current_model, explainer_type='tree')
            else:
                explainer = SHAPExplainer(current_model, explainer_type='tree')
            result = explainer.explain_instance(instance)
        
        elif explainer_type == 'lime':
            explainer = LIMEExplainer(current_model, mode=current_task)
            explainer.fit(current_data['X_train'])
            result = explainer.explain_instance(instance)
        
        # Get prediction
        prediction = current_model.predict([instance])[0]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top 8 features
        top_indices = np.argsort(np.abs(result.attributions))[-8:][::-1]
        top_features = [result.feature_names[i] for i in top_indices]
        top_attributions = [result.attributions[i] for i in top_indices]
        top_values = [result.feature_values[i] for i in top_indices]
        
        # Create color map
        colors = ['red' if x < 0 else 'green' for x in top_attributions]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_attributions, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([f'{feat}\n(val: {val:.2f})' for feat, val in zip(top_features, top_values)])
        ax.set_xlabel('Attribution Score')
        ax.set_title(f'{explainer_type.upper()} Local Explanation\nPrediction: {prediction:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_attributions)):
            ax.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                   va='center', ha='left' if val >= 0 else 'right')
        
        plt.tight_layout()
        plot_data = plot_to_base64(fig)
        
        return jsonify({
            'success': True,
            'explainer_type': explainer_type,
            'prediction': float(prediction),
            'baseline': float(result.baseline) if result.baseline is not None else None,
            'plot': plot_data,
            'top_features': top_features,
            'top_attributions': [float(x) for x in top_attributions],
            'top_values': [float(x) for x in top_values]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/explain_global', methods=['POST'])
def api_explain_global():
    """Generate global explanations"""
    if current_model is None:
        return jsonify({'success': False, 'error': 'No model trained'})
    
    data = request.json
    explainer_type = data.get('explainer_type', 'permutation')
    
    try:
        X_test = current_data['X_test']
        y_test = current_data['y_test']
        
        if explainer_type == 'permutation':
            scoring = 'accuracy' if current_task == 'classification' else 'r2'
            explainer = PermutationImportanceExplainer(
                current_model, 
                scoring=scoring,
                n_repeats=5,
                random_state=42
            )
            result = explainer.explain_global(X_test, y_test)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Sort features by importance
            sorted_indices = np.argsort(result.attributions)[::-1]
            sorted_features = [result.feature_names[i] for i in sorted_indices]
            sorted_importances = [result.attributions[i] for i in sorted_indices]
            sorted_stds = [result.metadata['std_importance'][i] for i in sorted_indices]
            
            # Create bar plot with error bars
            bars = ax.bar(range(len(sorted_features)), sorted_importances, 
                         yerr=sorted_stds, capsize=5, alpha=0.7, color='skyblue')
            ax.set_xticks(range(len(sorted_features)))
            ax.set_xticklabels(sorted_features, rotation=45, ha='right')
            ax.set_ylabel('Importance Score')
            ax.set_title(f'Permutation Importance\nBaseline {scoring}: {result.metadata["baseline_score"]:.3f}')
            
            # Add value labels on bars
            for i, (bar, val, std) in enumerate(zip(bars, sorted_importances, sorted_stds)):
                ax.text(i, val + std + 0.001, f'{val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_data = plot_to_base64(fig)
            
            return jsonify({
                'success': True,
                'explainer_type': explainer_type,
                'baseline_score': float(result.metadata['baseline_score']),
                'plot': plot_data,
                'feature_names': sorted_features,
                'importances': [float(x) for x in sorted_importances],
                'std_errors': [float(x) for x in sorted_stds]
            })
            
        elif explainer_type == 'partial_dependence':
            explainer = PartialDependenceExplainer(current_model, grid_resolution=20)
            
            # Use the most important feature (from permutation importance)
            perm_explainer = PermutationImportanceExplainer(
                current_model, 
                scoring='accuracy' if current_task == 'classification' else 'r2',
                n_repeats=3
            )
            perm_result = perm_explainer.explain_global(X_test, y_test)
            most_important_idx = np.argmax(perm_result.attributions)
            
            result = explainer.explain_global(X_test, features=most_important_idx)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            grid = result.metadata['grid']
            values = result.metadata['partial_dependence_values']
            
            ax.plot(grid, values, 'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel(f'{result.feature_names[0]} Value')
            ax.set_ylabel('Predicted Output')
            ax.set_title(f'Partial Dependence Plot\nFeature: {result.feature_names[0]}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_data = plot_to_base64(fig)
            
            return jsonify({
                'success': True,
                'explainer_type': explainer_type,
                'feature_name': result.feature_names[0],
                'plot': plot_data,
                'grid_values': [float(x) for x in grid],
                'pd_values': [float(x) for x in values]
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model_info')
def api_model_info():
    """Get current model information"""
    if current_model is None:
        return jsonify({'success': False, 'error': 'No model trained'})
    
    return jsonify({
        'success': True,
        'model_type': type(current_model).__name__,
        'task_type': current_task,
        'feature_names': feature_names,
        'data_shape': current_data['X_test'].shape if current_data else None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)