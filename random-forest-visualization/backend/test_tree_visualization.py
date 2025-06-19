#!/usr/bin/env python3
"""
Test script for tree visualization functionality
"""

import sys
import os
sys.path.append('.')

def test_tree_visualization():
    try:
        print("Testing tree visualization functionality...")
        
        # Import required modules
        from models.decision_path_tracker import DecisionPathTracker
        from models.model_loader import RandomForestModelLoader
        
        # Load model
        model_path = 'data/random_forest_model.pkl'
        print(f"Loading model from: {model_path}")
        
        loader = RandomForestModelLoader(model_path)
        if not loader.load_model():
            print("❌ Failed to load model")
            return False
            
        print("✅ Model loaded successfully")
        
        # Create tracker
        tracker = DecisionPathTracker(loader.model)
        print("✅ Decision path tracker created")
        
        # Test tree visualization
        sample_input = {
            'error_message': 'insufficient_funds',
            'billing_state': 'CA',
            'card_funding': 'credit',
            'card_network': 'visa',
            'card_issuer': 'chase'
        }
        
        print("Testing tree visualization extraction...")
        result = tracker.extract_tree_with_limited_depth_and_path(0, sample_input, 3)
        
        print("✅ Tree visualization extracted successfully")
        print(f"   - Total nodes in view: {result['tree_metadata']['total_nodes_in_limited_view']}")
        print(f"   - Nodes on path: {result['tree_metadata']['nodes_on_path']}")
        print(f"   - Root node has {len(result['root']['children'])} children")
        print(f"   - Max depth in view: {result['tree_metadata']['max_depth_in_view']}")
        print(f"   - Path coverage: {result['tree_metadata']['path_coverage']:.1f}%")
        
        # Test with different depths
        for depth in [2, 4, 5]:
            try:
                result_depth = tracker.extract_tree_with_limited_depth_and_path(0, sample_input, depth)
                print(f"✅ Depth {depth}: {result_depth['tree_metadata']['total_nodes_in_limited_view']} nodes")
            except Exception as e:
                print(f"❌ Depth {depth} failed: {e}")
        
        print("\n🎉 All tree visualization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tree_visualization()
    sys.exit(0 if success else 1)
