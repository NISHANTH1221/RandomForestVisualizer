#!/usr/bin/env python3
"""
Simple Model Analysis Script
Analyzes the Random Forest model without requiring additional imports
"""

import pickle
import os

def analyze_model():
    """Analyze the Random Forest model structure"""
    
    print("=== RANDOM FOREST MODEL ANALYSIS ===")
    
    try:
        # Load the model
        model_path = os.path.join(os.path.dirname(__file__), "data", "random_forest_model.pkl")
        print(f"Loading model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded successfully!")
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
            
            # Count leaf nodes manually
            leaf_count = 0
            for j in range(tree.node_count):
                if tree.children_left[j] == -1:  # This is a leaf node
                    leaf_count += 1
            
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
            
            # Count leaves
            leaf_count = 0
            for j in range(tree.node_count):
                if tree.children_left[j] == -1:
                    leaf_count += 1
            leaf_counts.append(leaf_count)
        
        # Calculate statistics manually
        def calculate_stats(values):
            if not values:
                return 0, 0, 0
            return min(values), max(values), sum(values) / len(values)
        
        min_nodes, max_nodes, avg_nodes = calculate_stats(node_counts)
        min_depth, max_depth, avg_depth = calculate_stats(depths)
        min_leaves, max_leaves, avg_leaves = calculate_stats(leaf_counts)
        
        print(f"Node count - Min: {min_nodes}, Max: {max_nodes}, Avg: {avg_nodes:.1f}")
        print(f"Tree depth - Min: {min_depth}, Max: {max_depth}, Avg: {avg_depth:.1f}")
        print(f"Leaf count - Min: {min_leaves}, Max: {max_leaves}, Avg: {avg_leaves:.1f}")
        
        # Count trees with many nodes
        large_trees_500 = sum(1 for n in node_counts if n > 500)
        large_trees_1000 = sum(1 for n in node_counts if n > 1000)
        large_trees_2000 = sum(1 for n in node_counts if n > 2000)
        
        print(f"\nTrees with >500 nodes: {large_trees_500}")
        print(f"Trees with >1000 nodes: {large_trees_1000}")
        print(f"Trees with >2000 nodes: {large_trees_2000}")
        
        # Analysis and warnings
        print("\n=== OVERFITTING ANALYSIS ===")
        
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
        print(f"- Max nodes in any tree: {max_nodes}")
        
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
        
        print("\n=== DECISION PATH IMPLICATIONS ===")
        print("For decision path visualization:")
        print(f"- Each decision path will have {avg_depth:.0f} nodes on average")
        print(f"- Longest possible path: {max_depth} nodes")
        print("- This is why you're seeing many nodes in decision paths")
        print("- The 2000+ nodes you mentioned is likely the TOTAL tree size")
        print("- But decision paths only show the actual path taken (much smaller)")
        
        print("\nAnalysis complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_model()
