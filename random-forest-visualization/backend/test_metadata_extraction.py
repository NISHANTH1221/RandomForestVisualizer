"""
Test script for metadata extraction with all 100 trees

This script implements T1.4.6 - Test metadata extraction with all 100 trees
It validates that all metadata extraction functions work correctly with the full Random Forest model.
"""

import time
import json
from pathlib import Path
from models.model_loader import (
    get_tree_metadata_json,
    get_tree_count_and_stats,
    get_tree_depths,
    get_feature_importance_for_trees,
    get_node_counts_per_tree,
    load_random_forest_model
)

def test_individual_functions():
    """Test each metadata extraction function individually"""
    print("=" * 60)
    print("TESTING INDIVIDUAL METADATA EXTRACTION FUNCTIONS")
    print("=" * 60)
    
    model_path = "data/random_forest_model.pkl"
    
    # Test 1: Tree count and stats
    print("\n1. Testing tree count and basic stats extraction...")
    start_time = time.time()
    tree_stats = get_tree_count_and_stats(model_path)
    elapsed = time.time() - start_time
    
    if "error" in tree_stats:
        print(f"❌ FAILED: {tree_stats['error']}")
        return False
    else:
        print(f"✅ SUCCESS: Analyzed {tree_stats['trees_analyzed']}/{tree_stats['total_trees']} trees in {elapsed:.2f}s")
        print(f"   - Success rate: {tree_stats['summary']['analysis_success_rate']:.2f}%")
        print(f"   - Average depth: {tree_stats['summary']['avg_depth']}")
        print(f"   - Average nodes: {tree_stats['summary']['avg_nodes']}")
    
    # Test 2: Tree depths
    print("\n2. Testing tree depth calculations...")
    start_time = time.time()
    depth_info = get_tree_depths(model_path)
    elapsed = time.time() - start_time
    
    if "error" in depth_info:
        print(f"❌ FAILED: {depth_info['error']}")
        return False
    else:
        print(f"✅ SUCCESS: Processed {depth_info['trees_processed']}/{depth_info['total_trees']} trees in {elapsed:.2f}s")
        print(f"   - Depth range: {depth_info['depth_statistics']['min_depth']}-{depth_info['depth_statistics']['max_depth']}")
        print(f"   - Average depth: {depth_info['depth_statistics']['avg_depth']}")
        print(f"   - Most common category: {max(depth_info['depth_categories'].items(), key=lambda x: x[1])[0]}")
    
    # Test 3: Feature importance
    print("\n3. Testing feature importance extraction...")
    start_time = time.time()
    importance_info = get_feature_importance_for_trees(model_path)
    elapsed = time.time() - start_time
    
    if "error" in importance_info:
        print(f"❌ FAILED: {importance_info['error']}")
        return False
    else:
        print(f"✅ SUCCESS: Processed {importance_info['trees_processed']}/{importance_info['total_trees']} trees in {elapsed:.2f}s")
        top_features = list(importance_info['importance_statistics']['most_important_features'].keys())[:3]
        print(f"   - Top 3 features: {top_features}")
        print(f"   - Errors: {len(importance_info['errors'])}")
    
    # Test 4: Node counts
    print("\n4. Testing node count calculations...")
    start_time = time.time()
    node_info = get_node_counts_per_tree(model_path)
    elapsed = time.time() - start_time
    
    if "error" in node_info:
        print(f"❌ FAILED: {node_info['error']}")
        return False
    else:
        print(f"✅ SUCCESS: Processed {node_info['trees_processed']}/{node_info['total_trees']} trees in {elapsed:.2f}s")
        print(f"   - Average nodes: {node_info['node_statistics']['avg_nodes']}")
        print(f"   - Total nodes in forest: {node_info['node_statistics']['total_nodes_all_trees']}")
        print(f"   - Internal to leaf ratio: {node_info['summary']['internal_to_leaf_ratio']}")
    
    return True

def test_unified_metadata_json():
    """Test the unified metadata JSON extraction"""
    print("\n" + "=" * 60)
    print("TESTING UNIFIED METADATA JSON EXTRACTION")
    print("=" * 60)
    
    model_path = "data/random_forest_model.pkl"
    
    print("\n5. Testing comprehensive metadata JSON creation...")
    start_time = time.time()
    metadata = get_tree_metadata_json(model_path)
    elapsed = time.time() - start_time
    
    if "error" in metadata:
        print(f"❌ FAILED: {metadata['error']}")
        return False
    
    print(f"✅ SUCCESS: Created unified metadata in {elapsed:.2f}s")
    
    # Validate structure
    expected_sections = [
        "model_info", "forest_statistics", "depth_analysis", 
        "node_analysis", "feature_importance", "individual_trees", 
        "generation_info", "summary"
    ]
    
    missing_sections = [section for section in expected_sections if section not in metadata]
    if missing_sections:
        print(f"❌ MISSING SECTIONS: {missing_sections}")
        return False
    
    print(f"✅ All required sections present: {len(expected_sections)}")
    
    # Validate individual trees data
    individual_trees = metadata.get("individual_trees", [])
    if len(individual_trees) != 100:
        print(f"❌ INCORRECT TREE COUNT: Expected 100, got {len(individual_trees)}")
        return False
    
    print(f"✅ All 100 trees have metadata")
    
    # Check data completeness
    completeness_rate = metadata["summary"]["data_completeness_rate"]
    if completeness_rate < 95.0:
        print(f"⚠️  LOW COMPLETENESS RATE: {completeness_rate:.2f}%")
    else:
        print(f"✅ High data completeness: {completeness_rate:.2f}%")
    
    # Validate individual tree structure
    sample_tree = individual_trees[0]
    required_tree_fields = ["tree_id", "basic_info", "categorization", "metrics", "feature_importance"]
    missing_tree_fields = [field for field in required_tree_fields if field not in sample_tree]
    
    if missing_tree_fields:
        print(f"❌ MISSING TREE FIELDS: {missing_tree_fields}")
        return False
    
    print(f"✅ Individual tree structure validated")
    
    return True

def test_data_consistency():
    """Test data consistency across different extraction methods"""
    print("\n" + "=" * 60)
    print("TESTING DATA CONSISTENCY")
    print("=" * 60)
    
    model_path = "data/random_forest_model.pkl"
    
    print("\n6. Testing data consistency across extraction methods...")
    
    # Get data from individual functions
    tree_stats = get_tree_count_and_stats(model_path)
    depth_info = get_tree_depths(model_path)
    node_info = get_node_counts_per_tree(model_path)
    
    # Get unified metadata
    metadata = get_tree_metadata_json(model_path)
    
    if any("error" in data for data in [tree_stats, depth_info, node_info, metadata]):
        print("❌ FAILED: Error in one or more extraction methods")
        return False
    
    # Check tree count consistency
    tree_counts = [
        tree_stats["total_trees"],
        depth_info["total_trees"],
        node_info["total_trees"],
        metadata["forest_statistics"]["total_trees"]
    ]
    
    if not all(count == tree_counts[0] for count in tree_counts):
        print(f"❌ INCONSISTENT TREE COUNTS: {tree_counts}")
        return False
    
    print(f"✅ Consistent tree count: {tree_counts[0]}")
    
    # Check depth consistency
    avg_depths = [
        tree_stats["summary"]["avg_depth"],
        depth_info["depth_statistics"]["avg_depth"],
        metadata["forest_statistics"]["avg_depth"]
    ]
    
    if not all(abs(depth - avg_depths[0]) < 0.01 for depth in avg_depths):
        print(f"❌ INCONSISTENT AVERAGE DEPTHS: {avg_depths}")
        return False
    
    print(f"✅ Consistent average depth: {avg_depths[0]}")
    
    # Check node count consistency
    avg_nodes = [
        tree_stats["summary"]["avg_nodes"],
        node_info["node_statistics"]["avg_nodes"],
        metadata["forest_statistics"]["avg_nodes"]
    ]
    
    if not all(abs(nodes - avg_nodes[0]) < 0.01 for nodes in avg_nodes):
        print(f"❌ INCONSISTENT AVERAGE NODES: {avg_nodes}")
        return False
    
    print(f"✅ Consistent average nodes: {avg_nodes[0]}")
    
    return True

def test_performance_benchmarks():
    """Test performance benchmarks for metadata extraction"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    model_path = "data/random_forest_model.pkl"
    
    print("\n7. Testing performance benchmarks...")
    
    # Benchmark individual functions
    functions_to_test = [
        ("Tree Stats", get_tree_count_and_stats),
        ("Tree Depths", get_tree_depths),
        ("Feature Importance", get_feature_importance_for_trees),
        ("Node Counts", get_node_counts_per_tree),
        ("Unified Metadata", get_tree_metadata_json)
    ]
    
    performance_results = {}
    
    for func_name, func in functions_to_test:
        print(f"\n   Benchmarking {func_name}...")
        start_time = time.time()
        result = func(model_path)
        elapsed = time.time() - start_time
        
        if "error" in result:
            print(f"   ❌ {func_name} failed: {result['error']}")
            return False
        
        performance_results[func_name] = elapsed
        print(f"   ✅ {func_name}: {elapsed:.2f}s")
    
    # Performance thresholds (reasonable for 100 trees)
    thresholds = {
        "Tree Stats": 30.0,      # 30 seconds max
        "Tree Depths": 15.0,     # 15 seconds max
        "Feature Importance": 20.0, # 20 seconds max
        "Node Counts": 15.0,     # 15 seconds max
        "Unified Metadata": 60.0  # 60 seconds max (calls all others)
    }
    
    print(f"\n   Performance Summary:")
    all_within_threshold = True
    
    for func_name, elapsed in performance_results.items():
        threshold = thresholds[func_name]
        status = "✅" if elapsed <= threshold else "⚠️"
        if elapsed > threshold:
            all_within_threshold = False
        print(f"   {status} {func_name}: {elapsed:.2f}s (threshold: {threshold}s)")
    
    if all_within_threshold:
        print(f"\n✅ All functions within performance thresholds")
    else:
        print(f"\n⚠️  Some functions exceeded performance thresholds")
    
    return True

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)
    
    print("\n8. Testing error handling with invalid inputs...")
    
    # Test with non-existent file
    print("   Testing with non-existent model file...")
    try:
        result = get_tree_metadata_json("nonexistent_model.pkl")
        if "error" in result:
            print("   ✅ Properly handled non-existent file")
        else:
            print("   ❌ Failed to handle non-existent file")
            return False
    except Exception as e:
        print(f"   ✅ Exception properly raised: {str(e)}")
    
    # Test with invalid model path
    print("   Testing with invalid model path...")
    try:
        result = get_tree_metadata_json("")
        if "error" in result:
            print("   ✅ Properly handled empty path")
        else:
            print("   ❌ Failed to handle empty path")
            return False
    except Exception as e:
        print(f"   ✅ Exception properly raised: {str(e)}")
    
    return True

def save_test_results(metadata):
    """Save test results to a JSON file for inspection"""
    print("\n" + "=" * 60)
    print("SAVING TEST RESULTS")
    print("=" * 60)
    
    output_file = "test_metadata_results.json"
    
    try:
        # Add timestamp to metadata
        metadata["generation_info"]["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        file_size = Path(output_file).stat().st_size / 1024  # KB
        print(f"✅ Test results saved to {output_file} ({file_size:.1f} KB)")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"- Total trees: {metadata['forest_statistics']['total_trees']}")
        print(f"- Data completeness: {metadata['summary']['data_completeness_rate']:.2f}%")
        print(f"- Most common depth category: {metadata['summary']['most_common_depth_category']}")
        print(f"- Most common node category: {metadata['summary']['most_common_node_category']}")
        print(f"- Average features per tree: {metadata['summary']['feature_diversity']['avg_features_per_tree']:.2f}")
        print(f"- Total errors: {metadata['generation_info']['errors']['total_errors']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save test results: {str(e)}")
        return False

def main():
    """Main test function"""
    print("RANDOM FOREST METADATA EXTRACTION - COMPREHENSIVE TEST")
    print("=" * 60)
    print("Testing metadata extraction with all 100 trees")
    print("This implements T1.4.6 - Test metadata extraction with all 100 trees")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        test_individual_functions,
        test_unified_metadata_json,
        test_data_consistency,
        test_performance_benchmarks,
        test_error_handling
    ]
    
    test_results = []
    for test_func in tests:
        try:
            result = test_func()
            test_results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {str(e)}")
            test_results.append(False)
    
    # Final metadata extraction for saving
    print("\n" + "=" * 60)
    print("FINAL METADATA EXTRACTION")
    print("=" * 60)
    
    final_metadata = get_tree_metadata_json("data/random_forest_model.pkl")
    if "error" not in final_metadata:
        save_test_results(final_metadata)
    
    # Summary
    total_time = time.time() - start_time
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Metadata extraction is working correctly with all 100 trees.")
        return True
    else:
        print(f"\n❌ {total_tests - passed_tests} test(s) failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
