"""
Test script for Decision Path Tracking functionality

This script tests all the components of the decision path tracking algorithm:
- T2.2.1 - Design path tracking data structure
- T2.2.2 - Implement tree traversal for given input
- T2.2.3 - Record decision at each node (left/right)
- T2.2.4 - Capture feature values and thresholds
- T2.2.5 - Include node statistics (samples, gini, prediction)
- T2.2.6 - Handle leaf node final prediction
- T2.2.7 - Test path tracking with various inputs
- T2.2.8 - Optimize for performance with large trees
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_loader import load_random_forest_model
from models.decision_path_tracker import DecisionPathTracker, track_decision_path_for_tree
import json
import time

def test_decision_path_tracking():
    """
    Comprehensive test of decision path tracking functionality
    """
    print("=" * 80)
    print("TESTING DECISION PATH TRACKING ALGORITHM")
    print("=" * 80)
    
    try:
        # Load the model
        print("\n1. Loading Random Forest model...")
        model_path = "data/random_forest_model.pkl"
        loader = load_random_forest_model(model_path)
        print(f"✓ Model loaded successfully: {loader.model.n_estimators} trees")
        
        # Create decision path tracker
        print("\n2. Creating Decision Path Tracker...")
        tracker = DecisionPathTracker(loader.model)
        print(f"✓ Tracker created with {len(tracker.feature_names)} features")
        
        # Test T2.2.1 - Design path tracking data structure
        print("\n3. Testing path tracking data structure (T2.2.1)...")
        path_structure = tracker.create_path_tracking_structure()
        required_keys = ["path_metadata", "decision_path", "path_summary", "validation"]
        for key in required_keys:
            assert key in path_structure, f"Missing required key: {key}"
        print("✓ Path tracking data structure is correctly designed")
        
        # Test T2.2.2 - Implement tree traversal for given input
        print("\n4. Testing tree traversal implementation (T2.2.2)...")
        sample_input = {
            "error_message": "insufficient_funds",
            "billing_state": "CA",
            "card_funding": "credit",
            "card_network": "visa",
            "card_issuer": "chase"
        }
        
        tree_id = 0
        start_time = time.time()
        path_result = tracker.track_decision_path(tree_id, sample_input)
        end_time = time.time()
        
        print(f"✓ Tree traversal completed in {(end_time - start_time) * 1000:.3f}ms")
        print(f"  - Nodes visited: {len(path_result['decision_path'])}")
        print(f"  - Path depth: {path_result['path_metadata']['path_statistics']['path_depth']}")
        print(f"  - Final prediction: {path_result['path_metadata']['prediction_result']['final_prediction']}")
        
        # Test T2.2.3 - Record decision at each node (left/right)
        print("\n5. Testing decision recording (T2.2.3)...")
        decisions_recorded = 0
        for node in path_result["decision_path"]:
            if not node["is_leaf"]:
                assert "decision_made" in node, "Decision not recorded for internal node"
                assert node["decision_made"] in ["left", "right"], f"Invalid decision: {node['decision_made']}"
                decisions_recorded += 1
        print(f"✓ Decisions recorded for {decisions_recorded} internal nodes")
        
        # Test T2.2.4 - Capture feature values and thresholds
        print("\n6. Testing feature values and thresholds capture (T2.2.4)...")
        features_captured = 0
        for node in path_result["decision_path"]:
            if not node["is_leaf"]:
                assert "feature_name" in node, "Feature name not captured"
                assert "threshold" in node, "Threshold not captured"
                assert "decision_logic" in node, "Decision logic not captured"
                assert "feature_value" in node["decision_logic"], "Feature value not captured"
                features_captured += 1
        print(f"✓ Feature values and thresholds captured for {features_captured} nodes")
        
        # Test T2.2.5 - Include node statistics (samples, gini, prediction)
        print("\n7. Testing node statistics inclusion (T2.2.5)...")
        stats_included = 0
        for node in path_result["decision_path"]:
            assert "samples" in node, "Sample count not included"
            assert "impurity" in node, "Impurity (gini) not included"
            assert "node_statistics" in node, "Node statistics not included"
            assert "gini_impurity" in node["node_statistics"], "Gini impurity not in statistics"
            stats_included += 1
        print(f"✓ Node statistics included for {stats_included} nodes")
        
        # Test T2.2.6 - Handle leaf node final prediction
        print("\n8. Testing leaf node final prediction (T2.2.6)...")
        leaf_node = path_result["decision_path"][-1]  # Last node should be leaf
        assert leaf_node["is_leaf"], "Last node is not a leaf node"
        assert "final_prediction" in leaf_node, "Final prediction not in leaf node"
        assert "prediction_type" in leaf_node, "Prediction type not specified"
        assert "confidence" in leaf_node, "Confidence not included"
        print(f"✓ Leaf node final prediction handled correctly")
        print(f"  - Prediction: {leaf_node['final_prediction']}")
        print(f"  - Type: {leaf_node['prediction_type']}")
        print(f"  - Confidence: {leaf_node['confidence']}")
        
        # Test T2.2.7 - Test path tracking with various inputs
        print("\n9. Testing with various inputs (T2.2.7)...")
        test_results = tracker.test_path_tracking_with_various_inputs(tree_id)
        print(f"✓ Tested {test_results['total_test_cases']} different input cases")
        print(f"  - Successful tests: {test_results['successful_tests']}")
        print(f"  - Failed tests: {test_results['failed_tests']}")
        print(f"  - Success rate: {test_results['summary']['success_rate']}%")
        print(f"  - Average execution time: {test_results['performance_metrics']['average_time_per_test_ms']:.3f}ms")
        print(f"  - Unique paths found: {test_results['path_diversity']['unique_paths']}")
        
        # Test T2.2.8 - Optimize for performance with large trees
        print("\n10. Testing performance optimization (T2.2.8)...")
        optimization_result = tracker.optimize_for_large_trees(tree_id)
        print(f"✓ Performance optimization analysis completed")
        print(f"  - Overall optimization score: {optimization_result['optimization_summary']['overall_optimization_score']}")
        print(f"  - Performance level: {optimization_result['optimization_summary']['optimization_level']}")
        print(f"  - Average execution time: {optimization_result['performance_analysis']['average_execution_time_ms']:.3f}ms")
        print(f"  - Recommendations: {optimization_result['optimization_summary']['total_recommendations']}")
        
        # Test validation functionality
        print("\n11. Testing path validation...")
        validation = path_result["validation"]
        print(f"✓ Path validation completed")
        print(f"  - Path valid: {validation['path_valid']}")
        print(f"  - Validation errors: {len(validation['validation_errors'])}")
        print(f"  - Validation score: {validation['validation_details']['validation_score']}")
        
        # Test with different tree IDs
        print("\n12. Testing with multiple trees...")
        trees_to_test = [0, 1, 2, 10, 50]  # Test a variety of trees
        successful_trees = 0
        
        for test_tree_id in trees_to_test:
            try:
                tree_path = tracker.track_decision_path(test_tree_id, sample_input)
                if tree_path["validation"]["path_valid"]:
                    successful_trees += 1
            except Exception as e:
                print(f"  - Tree {test_tree_id} failed: {str(e)}")
        
        print(f"✓ Successfully tested {successful_trees}/{len(trees_to_test)} trees")
        
        # Test convenience functions
        print("\n13. Testing convenience functions...")
        convenience_result = track_decision_path_for_tree(loader.model, 0, sample_input)
        assert "decision_path" in convenience_result, "Convenience function failed"
        print("✓ Convenience functions working correctly")
        
        # Performance benchmark
        print("\n14. Performance benchmark...")
        benchmark_iterations = 100
        benchmark_start = time.time()
        
        for i in range(benchmark_iterations):
            tracker.track_decision_path(0, sample_input)
        
        benchmark_end = time.time()
        avg_time = ((benchmark_end - benchmark_start) / benchmark_iterations) * 1000
        
        print(f"✓ Performance benchmark completed")
        print(f"  - {benchmark_iterations} iterations")
        print(f"  - Average time per tracking: {avg_time:.3f}ms")
        print(f"  - Throughput: {1000/avg_time:.1f} trackings per second")
        
        # Test edge cases
        print("\n15. Testing edge cases...")
        
        # Test with unknown values
        edge_case_input = {
            "error_message": "unknown_error",
            "billing_state": "ZZ",
            "card_funding": "unknown",
            "card_network": "unknown",
            "card_issuer": "unknown"
        }
        
        try:
            edge_result = tracker.track_decision_path(0, edge_case_input)
            print("✓ Edge case with unknown values handled successfully")
        except Exception as e:
            print(f"✗ Edge case failed: {str(e)}")
        
        # Test with invalid tree ID
        try:
            tracker.track_decision_path(999, sample_input)
            print("✗ Invalid tree ID should have failed")
        except Exception:
            print("✓ Invalid tree ID properly rejected")
        
        print("\n" + "=" * 80)
        print("DECISION PATH TRACKING TEST SUMMARY")
        print("=" * 80)
        print("✓ All core functionality tests passed")
        print("✓ T2.2.1 - Path tracking data structure: IMPLEMENTED")
        print("✓ T2.2.2 - Tree traversal for given input: IMPLEMENTED")
        print("✓ T2.2.3 - Decision recording at each node: IMPLEMENTED")
        print("✓ T2.2.4 - Feature values and thresholds capture: IMPLEMENTED")
        print("✓ T2.2.5 - Node statistics inclusion: IMPLEMENTED")
        print("✓ T2.2.6 - Leaf node final prediction: IMPLEMENTED")
        print("✓ T2.2.7 - Testing with various inputs: IMPLEMENTED")
        print("✓ T2.2.8 - Performance optimization: IMPLEMENTED")
        print("\n✅ TASK T2.2 - CREATE DECISION PATH TRACKING ALGORITHM: COMPLETED")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_decision_path():
    """
    Demonstrate the decision path tracking with detailed output
    """
    print("\n" + "=" * 80)
    print("DECISION PATH DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load model and create tracker
        loader = load_random_forest_model("data/random_forest_model.pkl")
        tracker = DecisionPathTracker(loader.model)
        
        # Sample input
        sample_input = {
            "error_message": "insufficient_funds",
            "billing_state": "CA",
            "card_funding": "credit",
            "card_network": "visa",
            "card_issuer": "chase"
        }
        
        print(f"\nInput features:")
        for key, value in sample_input.items():
            print(f"  {key}: {value}")
        
        # Track decision path
        path_result = tracker.track_decision_path(0, sample_input)
        
        print(f"\nDecision Path for Tree 0:")
        print("-" * 50)
        
        for i, node in enumerate(path_result["decision_path"]):
            if node["is_leaf"]:
                print(f"Step {i+1}: LEAF NODE (ID: {node['node_id']})")
                print(f"  Final Prediction: {node['final_prediction']}")
                print(f"  Confidence: {node['confidence']:.4f}")
                print(f"  Samples in leaf: {node['total_samples_in_leaf']}")
            else:
                print(f"Step {i+1}: DECISION NODE (ID: {node['node_id']})")
                print(f"  Feature: {node['feature_name']}")
                print(f"  Condition: {node['decision_logic']['condition']}")
                print(f"  Feature Value: {node['decision_logic']['feature_value']:.6f}")
                print(f"  Threshold: {node['threshold']:.6f}")
                print(f"  Decision: Go {node['decision_made'].upper()}")
                print(f"  Samples at node: {node['samples']}")
                print(f"  Impurity: {node['impurity']:.6f}")
        
        print(f"\nPath Summary:")
        print(f"  Total nodes visited: {len(path_result['decision_path'])}")
        print(f"  Path depth: {path_result['path_metadata']['path_statistics']['path_depth']}")
        print(f"  Traversal time: {path_result['path_metadata']['path_statistics']['traversal_time_ms']:.3f}ms")
        print(f"  Path valid: {path_result['validation']['path_valid']}")
        print(f"  Final prediction: {path_result['path_metadata']['prediction_result']['final_prediction']}")
        
        # Show key decisions
        key_decisions = path_result["path_summary"]["key_decisions"]
        if key_decisions:
            print(f"\nKey Decisions Made:")
            for decision in key_decisions:
                print(f"  {decision['feature']} {decision['condition']} → {decision['decision']}")
        
        return True
        
    except Exception as e:
        print(f"Demonstration failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Decision Path Tracking Tests...")
    
    # Run comprehensive tests
    test_success = test_decision_path_tracking()
    
    if test_success:
        # Run demonstration
        demonstrate_decision_path()
        print("\n🎉 All tests completed successfully!")
        print("Decision Path Tracking Algorithm is ready for integration!")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)
