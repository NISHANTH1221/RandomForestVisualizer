#!/usr/bin/env python3
"""
Model Analysis Script
Analyzes the Random Forest model to understand tree structure and node counts
"""

import pickle
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from model_loader import RandomForestModelLoader
    import numpy as np
    
    print("=== RANDOM FOREST MODEL ANALYSIS ===")
    
    # Load the model using our model loader
    model_path = os.path.join(os.path.dirname(__file__), "data", "random_forest_model.pkl")
    print(f"Loading model from: {model_path}")
    
    # Load with pickle first to check basic info
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Number of estimators: {model.n_estimators}")
    
    if hasattr(model, 'n_features_in_'):
        print(f"Number of features: {model.n_features_in_}")
    
    # Check model parameters
    print("\n=== MODEL PARAMETERS ===")
    params = model.get_params()
    relevant_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'max_features']
    for param in relevant_params:
        if param in params:
            print(f"{param}: {params[param]}")
    
    # Analyze first few trees
    print("\n=== TREE ANALYSIS (First 5 Trees) ===")
    for i in range(min(5, model.n_estimators)):
        tree = model.estimators_[i].tree_
        leaf_count = np.sum(tree.children_left == -1)
        internal_count = tree.node_count - leaf_count
        
        print(f"Tree {i}:")
        print(f"  - Total nodes: {tree.node_count}")
        print(f"  - Max depth: {tree.max_depth}")
        print(f"  - Leaf nodes: {leaf_count}")
        print(f"  - Internal nodes: {internal_count}")
        print(f"  - Samples at root: {tree.n_node_samples[0]}")
    
    # Analyze all trees to get statistics
    print("\n=== ALL TREES STATISTICS ===")
    node_counts = []
    depths = []
    leaf_counts = []
    
    for i in range(model.n_estimators):
        tree = model.estimators_[i].tree_
        node_counts.append(tree.node_count)
        depths.append(tree.max_depth)
        leaf_counts.append(np.sum(tree.children_left == -1))
    
    print(f"Node count - Min: {min(node_counts)}, Max: {max(node_counts)}, Avg: {np.mean(node_counts):.1f}")
    print(f"Tree depth - Min: {min(depths)}, Max: {max(depths)}, Avg: {np.mean(depths):.1f}")
    print(f"Leaf count - Min: {min(leaf_counts)}, Max: {max(leaf_counts)}, Avg: {np.mean(leaf_counts):.1f}")
    
    print(f"\nTrees with >500 nodes: {sum(1 for n in node_counts if n > 500)}")
    print(f"Trees with >1000 nodes: {sum(1 for n in node_counts if n > 1000)}")
    print(f"Trees with >2000 nodes: {sum(1 for n in node_counts if n > 2000)}")
    
    # Check if this indicates overfitting
    print("\n=== OVERFITTING ANALYSIS ===")
    avg_nodes = np.mean(node_counts)
    avg_depth = np.mean(depths)
    
    if avg_nodes > 1000:
        print("⚠️  WARNING: Trees are very large (avg >1000 nodes)")
        print("   This suggests potential overfitting")
        print("   Consider setting max_depth, min_samples_leaf, or max_leaf_nodes")
    
    if avg_depth > 20:
        print("⚠️  WARNING: Trees are very deep (avg >20 levels)")
        print("   This suggests potential overfitting")
        print("   Consider limiting max_depth")
    
    if params.get('max_depth') is None:
        print("⚠️  WARNING: max_depth is None (unlimited)")
        print("   Trees can grow arbitrarily deep")
    
    if params.get('min_samples_leaf', 1) == 1:
        print("⚠️  WARNING: min_samples_leaf is 1")
        print("   Trees can create leaves with single samples")
    
    # Explain why trees can have many nodes
    print("\n=== WHY TREES HAVE MANY NODES ===")
    print("In a binary tree, the number of nodes grows exponentially with depth:")
    print("- Depth 10: up to 2^11-1 = 2,047 nodes")
    print("- Depth 15: up to 2^16-1 = 65,535 nodes")
    print("- Depth 20: up to 2^21-1 = 2,097,151 nodes")
    print()
    print("Your trees have:")
    print(f"- Average depth: {avg_depth:.1f}")
    print(f"- Average nodes: {avg_nodes:.1f}")
    print(f"- Max nodes in any tree: {max(node_counts)}")
    
    # Check training data size
    if len(node_counts) > 0:
        first_tree = model.estimators_[0].tree_
        root_samples = first_tree.n_node_samples[0]
        print(f"\n=== TRAINING DATA ===")
        print(f"Samples at root (training data size): {root_samples}")
        
        if root_samples > 100000:
            print("Large training dataset can lead to deeper trees")
    
    print("\n=== RECOMMENDATIONS ===")
    if avg_nodes > 1000:
        print("1. Set max_depth=15 or max_depth=20 to limit tree growth")
        print("2. Set min_samples_leaf=5 or min_samples_leaf=10")
        print("3. Set max_leaf_nodes=1000 to directly limit tree size")
        print("4. Consider using max_features='sqrt' for feature subsampling")
    
    print("\nAnalysis complete!")

except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct pickle loading...")
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), "data", "random_forest_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded successfully: {type(model)}")
        print(f"Number of estimators: {model.n_estimators}")
        
        # Check first tree
        tree = model.estimators_[0].tree_
        print(f"First tree nodes: {tree.node_count}")
        print(f"First tree depth: {tree.max_depth}")
        
    except Exception as e2:
        print(f"Error loading model: {e2}")

except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()
