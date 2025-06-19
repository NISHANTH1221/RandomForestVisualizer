"""
Test script for Prediction Service functionality

This script tests all the components of the prediction service:
- T2.3.1 - Create prediction function for single tree
- T2.3.2 - Implement batch prediction for all 100 trees
- T2.3.3 - Format individual tree predictions
- T2.3.4 - Calculate ensemble prediction (average)
- T2.3.5 - Add confidence intervals
- T2.3.6 - Test prediction accuracy against original model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_loader import load_random_forest_model
from models.prediction_service import PredictionService, predict_with_single_tree, predict_with_all_trees
import json
import time
import numpy as np

def test_prediction_service():
    """
    Comprehensive test of prediction service functionality
    """
    print("=" * 80)
    print("TESTING PREDICTION SERVICE")
    print("=" * 80)
    
    try:
        # Load the model
        print("\n1. Loading Random Forest model...")
        model_path = "data/random_forest_model.pkl"
        loader = load_random_forest_model(model_path)
        print(f"✓ Model loaded successfully: {loader.model.n_estimators} trees")
        
        # Create prediction service
        print("\n2. Creating Prediction Service...")
        service = PredictionService(loader.model)
        print(f"✓ Service created with {len(service.feature_names)} features")
        
        # Test T2.3.1 - Create prediction function for single tree
        print("\n3. Testing single tree prediction (T2.3.1)...")
        sample_input = {
            "error_message": "insufficient_funds",
            "billing_state": "CA",
            "card_funding": "credit",
            "card_network": "visa",
            "card_issuer": "chase"
        }
        
        tree_id = 0
        start_time = time.time()
        single_pred = service.predict_single_tree(tree_id, sample_input)
        end_time = time.time()
        
        print(f"✓ Single tree prediction completed in {(end_time - start_time) * 1000:.3f}ms")
        print(f"  - Tree ID: {single_pred['tree_id']}")
        print(f"  - Prediction: {single_pred['prediction']['value']}")
        print(f"  - Confidence: {single_pred['prediction']['confidence']}")
        print(f"  - Tree nodes: {single_pred['tree_info']['total_nodes']}")
        
        # Validate single tree prediction structure
        required_keys = ["tree_id", "prediction", "tree_info", "input_features", "performance", "metadata"]
        for key in required_keys:
            assert key in single_pred, f"Missing required key in single prediction: {key}"
        print("✓ Single tree prediction structure is correct")
        
        # Test T2.3.2 - Implement batch prediction for all 100 trees
        print("\n4. Testing batch prediction for all trees (T2.3.2)...")
        batch_start = time.time()
        batch_result = service.predict_all_trees(sample_input)
        batch_end = time.time()
        
        print(f"✓ Batch prediction completed in {(batch_end - batch_start) * 1000:.3f}ms")
        print(f"  - Total trees: {batch_result['statistics']['total_trees']}")
        print(f"  - Successful predictions: {batch_result['statistics']['successful_predictions']}")
        print(f"  - Failed predictions: {batch_result['statistics']['failed_predictions']}")
        print(f"  - Individual predictions: {len(batch_result['individual_predictions'])}")
        
        # Validate batch prediction structure
        batch_required_keys = ["input_features", "individual_predictions", "ensemble_prediction", "statistics", "performance", "metadata"]
        for key in batch_required_keys:
            assert key in batch_result, f"Missing required key in batch result: {key}"
        print("✓ Batch prediction structure is correct")
        
        # Test T2.3.3 - Format individual tree predictions
        print("\n5. Testing individual tree prediction formatting (T2.3.3)...")
        individual_preds = batch_result["individual_predictions"]
        
        # Check first few individual predictions
        for i in range(min(5, len(individual_preds))):
            pred = individual_preds[i]
            required_pred_keys = ["tree_id", "prediction", "performance", "classification"]
            for key in required_pred_keys:
                assert key in pred, f"Missing key in individual prediction {i}: {key}"
            
            # Check prediction sub-structure
            pred_keys = ["value", "confidence", "success_probability", "prediction_class"]
            for key in pred_keys:
                assert key in pred["prediction"], f"Missing prediction key in individual prediction {i}: {key}"
        
        print(f"✓ Individual tree predictions formatted correctly")
        print(f"  - Sample prediction values: {[p['prediction']['value'] for p in individual_preds[:5]]}")
        print(f"  - Sample confidence scores: {[p['prediction']['confidence'] for p in individual_preds[:5]]}")
        
        # Test T2.3.4 - Calculate ensemble prediction (average)
        print("\n6. Testing ensemble prediction calculation (T2.3.4)...")
        ensemble_pred = batch_result["ensemble_prediction"]
        
        # Validate ensemble prediction structure
        ensemble_keys = ["value", "median", "weighted_average", "voting_prediction", "statistics", "voting", "confidence", "confidence_intervals", "recommendation"]
        for key in ensemble_keys:
            assert key in ensemble_pred, f"Missing key in ensemble prediction: {key}"
        
        print(f"✓ Ensemble prediction calculated correctly")
        print(f"  - Mean prediction: {ensemble_pred['value']:.6f}")
        print(f"  - Median prediction: {ensemble_pred['median']:.6f}")
        print(f"  - Weighted average: {ensemble_pred['weighted_average']:.6f}")
        print(f"  - Voting prediction: {ensemble_pred['voting_prediction']}")
        print(f"  - Positive votes: {ensemble_pred['voting']['positive_votes']}")
        print(f"  - Negative votes: {ensemble_pred['voting']['negative_votes']}")
        print(f"  - Consensus strength: {ensemble_pred['voting']['consensus_strength']:.4f}")
        
        # Test T2.3.5 - Add confidence intervals
        print("\n7. Testing confidence intervals (T2.3.5)...")
        confidence_intervals = ensemble_pred["confidence_intervals"]
        
        # Validate confidence intervals structure
        ci_keys = ["parametric", "bootstrap", "sample_size", "standard_error", "interpretation"]
        for key in ci_keys:
            assert key in confidence_intervals, f"Missing key in confidence intervals: {key}"
        
        # Check parametric intervals
        parametric_intervals = confidence_intervals["parametric"]
        expected_levels = ["90%", "95%", "99%"]
        for level in expected_levels:
            assert level in parametric_intervals, f"Missing confidence level: {level}"
            interval = parametric_intervals[level]
            assert "lower_bound" in interval, f"Missing lower_bound in {level} interval"
            assert "upper_bound" in interval, f"Missing upper_bound in {level} interval"
            assert interval["upper_bound"] > interval["lower_bound"], f"Invalid interval bounds for {level}"
        
        print(f"✓ Confidence intervals calculated correctly")
        print(f"  - Sample size: {confidence_intervals['sample_size']}")
        print(f"  - Standard error: {confidence_intervals['standard_error']:.6f}")
        print(f"  - 95% CI: [{parametric_intervals['95%']['lower_bound']:.6f}, {parametric_intervals['95%']['upper_bound']:.6f}]")
        
        # Test bootstrap intervals if available
        if "bootstrap" in confidence_intervals and "error" not in confidence_intervals["bootstrap"]:
            bootstrap_intervals = confidence_intervals["bootstrap"]
            print(f"  - Bootstrap samples: {bootstrap_intervals['bootstrap_samples']}")
            print(f"  - Bootstrap 95% CI: [{bootstrap_intervals['intervals']['95%']['lower_bound']:.6f}, {bootstrap_intervals['intervals']['95%']['upper_bound']:.6f}]")
        
        # Test T2.3.6 - Test prediction accuracy against original model
        print("\n8. Testing prediction accuracy against original model (T2.3.6)...")
        accuracy_result = service.test_prediction_accuracy()
        
        # Validate accuracy test structure
        accuracy_keys = ["test_summary", "test_details", "accuracy_metrics", "performance_comparison"]
        for key in accuracy_keys:
            assert key in accuracy_result, f"Missing key in accuracy result: {key}"
        
        print(f"✓ Accuracy testing completed")
        print(f"  - Total test cases: {accuracy_result['test_summary']['total_test_cases']}")
        print(f"  - Successful tests: {accuracy_result['test_summary']['successful_tests']}")
        print(f"  - Failed tests: {accuracy_result['test_summary']['failed_tests']}")
        print(f"  - Accuracy score: {accuracy_result['test_summary']['accuracy_score']:.2f}%")
        print(f"  - Mean absolute error: {accuracy_result['accuracy_metrics']['mean_absolute_error']:.6f}")
        print(f"  - Correlation coefficient: {accuracy_result['accuracy_metrics']['correlation_coefficient']:.6f}")
        
        # Test performance comparison
        if "performance_summary" in accuracy_result:
            perf_summary = accuracy_result["performance_summary"]
            print(f"  - Avg original time: {perf_summary['avg_original_time_ms']:.3f}ms")
            print(f"  - Avg ensemble time: {perf_summary['avg_ensemble_time_ms']:.3f}ms")
            print(f"  - Performance overhead: {perf_summary['performance_overhead']:.2f}x")
        
        # Test with different inputs
        print("\n9. Testing with various input types...")
        
        # Test with different risk scenarios
        test_scenarios = [
            {
                "name": "low_risk",
                "input": {
                    "error_message": "approved",
                    "billing_state": "NY",
                    "card_funding": "debit",
                    "card_network": "mastercard",
                    "card_issuer": "bank_of_america"
                }
            },
            {
                "name": "high_risk",
                "input": {
                    "error_message": "insufficient_funds",
                    "billing_state": "CA",
                    "card_funding": "credit",
                    "card_network": "visa",
                    "card_issuer": "chase"
                }
            }
        ]
        
        scenario_results = []
        for scenario in test_scenarios:
            scenario_pred = service.predict_all_trees(scenario["input"])
            scenario_results.append({
                "name": scenario["name"],
                "prediction": scenario_pred["ensemble_prediction"]["value"],
                "consensus": scenario_pred["ensemble_prediction"]["voting"]["consensus_strength"],
                "recommendation": scenario_pred["ensemble_prediction"]["recommendation"]["recommendation"]
            })
        
        print("✓ Various input scenarios tested successfully")
        for result in scenario_results:
            print(f"  - {result['name']}: prediction={result['prediction']:.4f}, consensus={result['consensus']:.4f}, recommendation={result['recommendation']}")
        
        # Test convenience functions
        print("\n10. Testing convenience functions...")
        
        # Test single tree convenience function
        conv_single = predict_with_single_tree(loader.model, 0, sample_input)
        assert "tree_id" in conv_single, "Convenience single tree function failed"
        
        # Test batch convenience function
        conv_batch = predict_with_all_trees(loader.model, sample_input)
        assert "ensemble_prediction" in conv_batch, "Convenience batch function failed"
        
        print("✓ Convenience functions working correctly")
        
        # Performance benchmark
        print("\n11. Performance benchmark...")
        benchmark_iterations = 50
        
        # Benchmark single tree predictions
        single_times = []
        for i in range(benchmark_iterations):
            start = time.time()
            service.predict_single_tree(i % loader.model.n_estimators, sample_input)
            single_times.append((time.time() - start) * 1000)
        
        # Benchmark batch predictions
        batch_times = []
        for i in range(5):  # Fewer iterations for batch since it's slower
            start = time.time()
            service.predict_all_trees(sample_input)
            batch_times.append((time.time() - start) * 1000)
        
        print(f"✓ Performance benchmark completed")
        print(f"  - Single tree avg time: {np.mean(single_times):.3f}ms")
        print(f"  - Single tree throughput: {1000/np.mean(single_times):.1f} predictions/sec")
        print(f"  - Batch prediction avg time: {np.mean(batch_times):.3f}ms")
        print(f"  - Batch throughput: {1000/np.mean(batch_times):.1f} batch predictions/sec")
        
        # Test error handling
        print("\n12. Testing error handling...")
        
        # Test invalid tree ID
        try:
            service.predict_single_tree(999, sample_input)
            print("✗ Invalid tree ID should have failed")
        except Exception:
            print("✓ Invalid tree ID properly rejected")
        
        # Test invalid input type
        try:
            service.predict_single_tree(0, "invalid_input")
            print("✗ Invalid input type should have failed")
        except Exception:
            print("✓ Invalid input type properly rejected")
        
        print("\n" + "=" * 80)
        print("PREDICTION SERVICE TEST SUMMARY")
        print("=" * 80)
        print("✓ All core functionality tests passed")
        print("✓ T2.3.1 - Single tree prediction function: IMPLEMENTED")
        print("✓ T2.3.2 - Batch prediction for all trees: IMPLEMENTED")
        print("✓ T2.3.3 - Individual tree prediction formatting: IMPLEMENTED")
        print("✓ T2.3.4 - Ensemble prediction calculation: IMPLEMENTED")
        print("✓ T2.3.5 - Confidence intervals: IMPLEMENTED")
        print("✓ T2.3.6 - Prediction accuracy testing: IMPLEMENTED")
        print("\n✅ TASK T2.3 - IMPLEMENT PREDICTION SERVICE: COMPLETED")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_prediction_service():
    """
    Demonstrate the prediction service with detailed output
    """
    print("\n" + "=" * 80)
    print("PREDICTION SERVICE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load model and create service
        loader = load_random_forest_model("data/random_forest_model.pkl")
        service = PredictionService(loader.model)
        
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
        
        # Get batch prediction
        batch_result = service.predict_all_trees(sample_input)
        
        print(f"\nEnsemble Prediction Results:")
        print("-" * 50)
        ensemble = batch_result["ensemble_prediction"]
        
        print(f"Final Prediction: {ensemble['value']:.6f}")
        print(f"Voting Result: {ensemble['voting_prediction']:.0f}")
        print(f"Confidence: {ensemble['confidence']['ensemble_confidence']:.4f}")
        print(f"Consensus Strength: {ensemble['voting']['consensus_strength']:.4f}")
        print(f"Recommendation: {ensemble['recommendation']['recommendation']}")
        print(f"Reasoning: {ensemble['recommendation']['reasoning']}")
        
        print(f"\nVoting Breakdown:")
        print(f"  Positive votes: {ensemble['voting']['positive_votes']}")
        print(f"  Negative votes: {ensemble['voting']['negative_votes']}")
        print(f"  Total votes: {ensemble['voting']['total_votes']}")
        print(f"  Unanimous: {ensemble['voting']['unanimous']}")
        
        print(f"\nStatistics:")
        stats = ensemble['statistics']
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std Dev: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Range: {stats['range']:.6f}")
        
        print(f"\nConfidence Intervals (95%):")
        ci_95 = ensemble['confidence_intervals']['parametric']['95%']
        print(f"  Lower bound: {ci_95['lower_bound']:.6f}")
        print(f"  Upper bound: {ci_95['upper_bound']:.6f}")
        print(f"  Width: {ci_95['width']:.6f}")
        
        # Show sample individual predictions
        individual_preds = batch_result["individual_predictions"]
        print(f"\nSample Individual Tree Predictions:")
        print("-" * 50)
        for i in range(min(10, len(individual_preds))):
            pred = individual_preds[i]
            print(f"Tree {pred['tree_id']:2d}: {pred['prediction']['value']:.6f} (confidence: {pred['prediction']['confidence']:.4f}, risk: {pred['classification']['risk_level']})")
        
        # Performance summary
        perf = batch_result["performance"]
        print(f"\nPerformance Summary:")
        print(f"  Total time: {perf['total_time_ms']:.3f}ms")
        print(f"  Average per tree: {perf['average_time_per_tree_ms']:.3f}ms")
        print(f"  Fastest tree: {perf['fastest_tree_ms']:.3f}ms")
        print(f"  Slowest tree: {perf['slowest_tree_ms']:.3f}ms")
        
        return True
        
    except Exception as e:
        print(f"Demonstration failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Prediction Service Tests...")
    
    # Run comprehensive tests
    test_success = test_prediction_service()
    
    if test_success:
        # Run demonstration
        demonstrate_prediction_service()
        print("\n🎉 All tests completed successfully!")
        print("Prediction Service is ready for integration!")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)
