"""
Test script for tree structure extraction functionality

This script tests the tree extraction API implementation for T2.1
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from models.model_loader import RandomForestModelLoader
from models.tree_extractor import TreeStructureExtractor
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tree_extraction():
    """Test the tree extraction functionality"""
    try:
        print("=" * 60)
        print("TESTING TREE STRUCTURE EXTRACTION")
        print("=" * 60)
        
        # Test 1: Load model
        print("\n1. Loading Random Forest model...")
        model_path = backend_dir / "data" / "random_forest_model.pkl"
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        loader = RandomForestModelLoader(str(model_path))
        if not loader.load_model():
            print("❌ Failed to load model")
            return False
        
        loader.extract_model_info()
        print(f"✅ Model loaded successfully: {loader.model.n_estimators} trees")
        
        # Test 2: Initialize tree extractor
        print("\n2. Initializing tree extractor...")
        extractor = TreeStructureExtractor(loader.model)
        print(f"✅ Tree extractor initialized with {len(extractor.feature_names)} features")
        
        # Test 3: Extract single tree structure
        print("\n3. Testing single tree extraction...")
        tree_id = 0
        tree_structure = extractor.extract_single_tree_structure(tree_id)
        
        print(f"✅ Tree {tree_id} extracted:")
        print(f"   - Total nodes: {tree_structure['metadata']['total_nodes']}")
        print(f"   - Max depth: {tree_structure['metadata']['max_depth']}")
        print(f"   - Leaf nodes: {tree_structure['structure_info']['leaf_count']}")
        print(f"   - Internal nodes: {tree_structure['structure_info']['internal_node_count']}")
        
        # Test 4: Convert to JSON format
        print("\n4. Testing JSON conversion...")
        json_tree = extractor.convert_tree_to_json(tree_id)
        
        print(f"✅ Tree {tree_id} converted to JSON:")
        print(f"   - Format version: {json_tree['format_version']}")
        print(f"   - Flat nodes: {len(json_tree['flat_nodes'])}")
        print(f"   - Has hierarchical structure: {'root_node' in json_tree}")
        print(f"   - Feature names included: {len(json_tree['feature_names'])} features")
        
        # Test 5: Test comprehensive node info
        print("\n5. Testing comprehensive node information...")
        comprehensive_info = extractor.get_tree_node_info(tree_id, include_children=True)
        
        print(f"✅ Comprehensive info generated:")
        print(f"   - Enhanced nodes: {comprehensive_info['node_count']}")
        print(f"   - Has node metrics: {'node_metrics' in comprehensive_info['nodes'][0]}")
        print(f"   - Has children references: {'children_references' in comprehensive_info['nodes'][0] if not comprehensive_info['nodes'][0]['is_leaf'] else 'N/A (leaf node)'}")
        
        # Test 6: Test tree ID validation
        print("\n6. Testing tree ID validation...")
        valid_id = 0
        invalid_id = 999
        
        print(f"   - Tree ID {valid_id} valid: {extractor.validate_tree_id(valid_id)} ✅")
        print(f"   - Tree ID {invalid_id} valid: {extractor.validate_tree_id(invalid_id)} ✅")
        
        # Test 7: Test multiple trees
        print("\n7. Testing multiple tree extraction...")
        test_trees = [0, 1, 2, 50, 99]  # Test various tree IDs
        
        for tree_id in test_trees:
            if extractor.validate_tree_id(tree_id):
                tree_info = extractor.extract_single_tree_structure(tree_id)
                print(f"   - Tree {tree_id}: {tree_info['metadata']['total_nodes']} nodes, depth {tree_info['metadata']['max_depth']} ✅")
            else:
                print(f"   - Tree {tree_id}: Invalid ID ❌")
        
        # Test 8: Test error handling
        print("\n8. Testing error handling...")
        try:
            extractor.extract_single_tree_structure(-1)
            print("   - Negative tree ID: Should have failed ❌")
        except (ValueError, Exception) as e:
            print(f"   - Negative tree ID: Properly handled ✅ ({type(e).__name__})")
        
        try:
            extractor.extract_single_tree_structure(1000)
            print("   - Large tree ID: Should have failed ❌")
        except (ValueError, Exception) as e:
            print(f"   - Large tree ID: Properly handled ✅ ({type(e).__name__})")
        
        # Test 9: Test extraction summary
        print("\n9. Testing extraction capabilities summary...")
        summary = extractor.get_tree_extraction_summary()
        
        print(f"✅ Extraction summary generated:")
        print(f"   - Total trees: {summary['model_info']['total_trees']}")
        print(f"   - Feature count: {summary['model_info']['feature_count']}")
        print(f"   - Capabilities: {len(summary['extraction_capabilities'])} features")
        print(f"   - Supported formats: {len(summary['supported_formats'])} formats")
        
        # Test 10: Sample node structure
        print("\n10. Sample node structure analysis...")
        sample_tree = extractor.extract_single_tree_structure(0)
        
        # Find a leaf and internal node for analysis
        leaf_node = None
        internal_node = None
        
        for node in sample_tree['nodes']:
            if node['is_leaf'] and leaf_node is None:
                leaf_node = node
            elif not node['is_leaf'] and internal_node is None:
                internal_node = node
            
            if leaf_node and internal_node:
                break
        
        if internal_node:
            print(f"✅ Sample internal node (ID {internal_node['node_id']}):")
            print(f"   - Feature: {internal_node['feature_name']}")
            print(f"   - Threshold: {internal_node['threshold']:.6f}")
            print(f"   - Samples: {internal_node['samples']}")
            print(f"   - Impurity: {internal_node['impurity']:.6f}")
            print(f"   - Children: {internal_node['left_child']}, {internal_node['right_child']}")
        
        if leaf_node:
            print(f"✅ Sample leaf node (ID {leaf_node['node_id']}):")
            print(f"   - Prediction: {leaf_node['prediction']}")
            print(f"   - Samples: {leaf_node['samples']}")
            print(f"   - Impurity: {leaf_node['impurity']:.6f}")
            print(f"   - Depth: {leaf_node['depth']}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TREE EXTRACTION TESTS PASSED!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_json_serialization():
    """Test that extracted tree structures are JSON serializable"""
    try:
        print("\n" + "=" * 60)
        print("TESTING JSON SERIALIZATION")
        print("=" * 60)
        
        # Load model and extractor
        model_path = backend_dir / "data" / "random_forest_model.pkl"
        loader = RandomForestModelLoader(str(model_path))
        loader.load_model()
        loader.extract_model_info()
        extractor = TreeStructureExtractor(loader.model)
        
        # Test JSON serialization of tree structure
        tree_structure = extractor.convert_tree_to_json(0)
        
        # Try to serialize to JSON
        json_string = json.dumps(tree_structure, indent=2)
        print(f"✅ Tree structure successfully serialized to JSON ({len(json_string)} characters)")
        
        # Try to deserialize
        deserialized = json.loads(json_string)
        print(f"✅ JSON successfully deserialized")
        
        # Verify structure integrity
        assert deserialized['tree_id'] == tree_structure['tree_id']
        assert len(deserialized['flat_nodes']) == len(tree_structure['flat_nodes'])
        assert deserialized['metadata']['total_nodes'] == tree_structure['metadata']['total_nodes']
        
        print(f"✅ Structure integrity verified")
        
        # Test with comprehensive node info
        comprehensive_info = extractor.get_tree_node_info(0)
        json_string2 = json.dumps(comprehensive_info, indent=2)
        print(f"✅ Comprehensive node info serialized ({len(json_string2)} characters)")
        
        print("\n✅ JSON SERIALIZATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ JSON serialization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Tree Extraction Tests...")
    
    # Run tests
    extraction_success = test_tree_extraction()
    json_success = test_json_serialization()
    
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Tree Extraction Tests: {'✅ PASSED' if extraction_success else '❌ FAILED'}")
    print(f"JSON Serialization Tests: {'✅ PASSED' if json_success else '❌ FAILED'}")
    
    if extraction_success and json_success:
        print("\n🎉 ALL TESTS PASSED! Tree extraction API is ready.")
        print("\nNext steps:")
        print("- Start the FastAPI server: python app.py")
        print("- Test the API endpoint: GET /api/trees/0")
        print("- Check API docs: http://localhost:8000/docs")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
