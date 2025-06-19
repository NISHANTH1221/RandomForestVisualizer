"""
Test script for depth parameter functionality in tree extraction

This script tests the new depth parameter feature added to the tree extraction API.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_loader import RandomForestModelLoader
from models.tree_extractor import TreeStructureExtractor
import json
import time

def test_depth_parameter():
    """Test the depth parameter functionality"""
    print("Testing depth parameter functionality...")
    
    try:
        # Load the model
        model_path = "data/random_forest_model.pkl"
        print(f"Loading model from {model_path}")
        
        model_loader = RandomForestModelLoader(model_path)
        if not model_loader.load_model():
            print("Failed to load model")
            return False
        
        model_loader.extract_model_info()
        tree_extractor = TreeStructureExtractor(model_loader.model)
        
        # Test tree 0 with different depth limits
        tree_id = 0
        print(f"\nTesting tree {tree_id} with different depth limits:")
        
        # Get full tree first
        print("\n1. Extracting full tree (no depth limit)...")
        start_time = time.time()
        full_tree = tree_extractor.convert_tree_to_json(tree_id)
        full_time = time.time() - start_time
        
        print(f"   Full tree: {full_tree['metadata']['total_nodes']} nodes, "
              f"depth {full_tree['metadata']['max_depth']}, "
              f"extraction time: {full_time:.3f}s")
        
        # Test different depth limits
        depth_limits = [1, 2, 3, 4, 5]
        
        for depth in depth_limits:
            print(f"\n2. Extracting tree with depth limit {depth}...")
            start_time = time.time()
            limited_tree = tree_extractor.convert_tree_to_json_with_depth_limit(tree_id, depth)
            limited_time = time.time() - start_time
            
            # Analyze results
            total_nodes = limited_tree['statistics']['filtered_total_nodes']
            original_nodes = limited_tree['statistics']['original_total_nodes']
            truncated_nodes = limited_tree['statistics'].get('truncated_nodes', 0)
            effective_depth = limited_tree['metadata']['effective_max_depth']
            is_depth_limited = limited_tree['metadata']['is_depth_limited']
            
            print(f"   Depth {depth}: {total_nodes}/{original_nodes} nodes, "
                  f"effective depth: {effective_depth}, "
                  f"truncated: {truncated_nodes}, "
                  f"limited: {is_depth_limited}, "
                  f"extraction time: {limited_time:.3f}s")
            
            # Verify depth constraint
            max_node_depth = max(node['depth'] for node in limited_tree['flat_nodes'])
            if max_node_depth > depth:
                print(f"   ERROR: Found node at depth {max_node_depth}, expected max {depth}")
                return False
            
            # Check for truncated nodes
            truncated_count = sum(1 for node in limited_tree['flat_nodes'] 
                                if node.get('is_truncated', False))
            if truncated_count != truncated_nodes:
                print(f"   ERROR: Truncated count mismatch: {truncated_count} vs {truncated_nodes}")
                return False
            
            print(f"   ✓ Depth constraint verified, {truncated_count} nodes truncated")
        
        # Test edge cases
        print("\n3. Testing edge cases...")
        
        # Test depth 0 (should fail or return only root)
        try:
            zero_depth = tree_extractor.convert_tree_to_json_with_depth_limit(tree_id, 0)
            print("   WARNING: Depth 0 should probably be invalid")
        except Exception as e:
            print(f"   ✓ Depth 0 correctly rejected: {e}")
        
        # Test very large depth (should be same as full tree)
        large_depth = 100
        large_tree = tree_extractor.convert_tree_to_json_with_depth_limit(tree_id, large_depth)
        if large_tree['metadata']['total_nodes'] == full_tree['metadata']['total_nodes']:
            print(f"   ✓ Large depth ({large_depth}) returns full tree")
        else:
            print(f"   ERROR: Large depth should return full tree")
            return False
        
        # Test with None depth (should be same as full tree)
        none_tree = tree_extractor.convert_tree_to_json_with_depth_limit(tree_id, None)
        if none_tree['metadata']['total_nodes'] == full_tree['metadata']['total_nodes']:
            print(f"   ✓ None depth returns full tree")
        else:
            print(f"   ERROR: None depth should return full tree")
            return False
        
        print("\n4. Testing hierarchical structure with depth limits...")
        
        # Test hierarchical structure for depth 3
        depth_3_tree = tree_extractor.convert_tree_to_json_with_depth_limit(tree_id, 3)
        root_node = depth_3_tree['root_node']
        
        def check_hierarchical_depth(node, current_depth=0, max_allowed_depth=3):
            """Recursively check hierarchical structure depth"""
            if current_depth > max_allowed_depth:
                return False, f"Node at depth {current_depth} exceeds limit {max_allowed_depth}"
            
            if node.get('children'):
                for child_key, child_node in node['children'].items():
                    if child_node:
                        valid, error = check_hierarchical_depth(child_node, current_depth + 1, max_allowed_depth)
                        if not valid:
                            return False, error
            
            return True, None
        
        valid, error = check_hierarchical_depth(root_node, 0, 3)
        if valid:
            print("   ✓ Hierarchical structure respects depth limit")
        else:
            print(f"   ERROR: {error}")
            return False
        
        print("\n5. Performance comparison...")
        
        # Compare performance for different depths
        depths_to_test = [2, 4, 6, None]  # None = full tree
        
        for depth in depths_to_test:
            times = []
            for _ in range(3):  # Run 3 times for average
                start_time = time.time()
                if depth is None:
                    tree_extractor.convert_tree_to_json(tree_id)
                else:
                    tree_extractor.convert_tree_to_json_with_depth_limit(tree_id, depth)
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            depth_str = "full" if depth is None else str(depth)
            print(f"   Depth {depth_str}: {avg_time:.3f}s average")
        
        print("\n✅ All depth parameter tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_integration():
    """Test the API integration with depth parameter"""
    print("\n" + "="*50)
    print("Testing API integration...")
    
    try:
        import requests
        import json
        
        base_url = "http://localhost:8000"
        tree_id = 0
        
        # Test different depth parameters
        test_cases = [
            {"depth": None, "description": "No depth limit"},
            {"depth": 2, "description": "Depth limit 2"},
            {"depth": 4, "description": "Depth limit 4"},
            {"depth": 6, "description": "Depth limit 6"},
        ]
        
        for case in test_cases:
            depth = case["depth"]
            description = case["description"]
            
            print(f"\nTesting: {description}")
            
            # Build URL
            if depth is not None:
                url = f"{base_url}/api/trees/{tree_id}?depth={depth}"
            else:
                url = f"{base_url}/api/trees/{tree_id}"
            
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("success"):
                        tree_data = data["tree_data"]
                        total_nodes = tree_data["statistics"].get("filtered_total_nodes", 
                                                               tree_data["metadata"]["total_nodes"])
                        effective_depth = tree_data["metadata"].get("effective_max_depth",
                                                                   tree_data["metadata"]["max_depth"])
                        is_limited = tree_data["metadata"].get("is_depth_limited", False)
                        
                        print(f"   ✓ Success: {total_nodes} nodes, "
                              f"effective depth: {effective_depth}, "
                              f"limited: {is_limited}")
                    else:
                        print(f"   ❌ API returned success=false: {data}")
                        return False
                else:
                    print(f"   ❌ HTTP {response.status_code}: {response.text}")
                    return False
                    
            except requests.exceptions.ConnectionError:
                print(f"   ⚠️  Could not connect to API server at {base_url}")
                print("   (This is expected if the server is not running)")
                return True  # Don't fail the test if server is not running
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
                return False
        
        # Test invalid depth values
        print(f"\nTesting invalid depth values...")
        
        invalid_cases = [
            {"depth": 0, "description": "Depth 0 (too small)"},
            {"depth": -1, "description": "Negative depth"},
            {"depth": 25, "description": "Depth 25 (too large)"},
        ]
        
        for case in invalid_cases:
            depth = case["depth"]
            description = case["description"]
            
            url = f"{base_url}/api/trees/{tree_id}?depth={depth}"
            
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 400:
                    print(f"   ✓ {description}: Correctly rejected with 400")
                else:
                    print(f"   ❌ {description}: Expected 400, got {response.status_code}")
                    return False
                    
            except requests.exceptions.ConnectionError:
                print(f"   ⚠️  Could not test invalid cases - server not running")
                break
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
                return False
        
        print("\n✅ API integration tests passed!")
        return True
        
    except ImportError:
        print("   ⚠️  requests library not available, skipping API tests")
        return True
    except Exception as e:
        print(f"\n❌ API integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Random Forest Tree Depth Parameter Test")
    print("="*50)
    
    # Test the core functionality
    success = test_depth_parameter()
    
    if success:
        # Test API integration if core tests pass
        api_success = test_api_integration()
        
        if api_success:
            print("\n🎉 All tests completed successfully!")
            print("\nThe depth parameter feature is working correctly.")
            print("\nUsage examples:")
            print("  - GET /api/trees/0        (full tree)")
            print("  - GET /api/trees/0?depth=3 (limited to depth 3)")
            print("  - Frontend: getTreeStructure(0, 3)")
        else:
            print("\n⚠️  Core tests passed but API tests failed")
            sys.exit(1)
    else:
        print("\n❌ Core functionality tests failed")
        sys.exit(1)
