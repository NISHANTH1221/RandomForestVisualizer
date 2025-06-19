"""
Tree Structure Extraction Module

This module handles extracting detailed tree structures from Random Forest models.
It provides functions to:
- Extract single tree structure from sklearn trees
- Convert sklearn tree to JSON format
- Include comprehensive node information (feature, threshold, samples, gini)
- Add children node references for tree traversal
- Handle error cases for invalid tree IDs
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import json

# Configure logging
logger = logging.getLogger(__name__)

class TreeStructureExtractor:
    """
    A class to handle extracting detailed tree structures from Random Forest models
    """
    
    def __init__(self, model):
        """
        Initialize the tree extractor with a Random Forest model
        
        Args:
            model: Trained Random Forest model (sklearn)
        """
        self.model = model
        self.feature_names = None
        
        # Extract feature names if available
        if hasattr(model, 'feature_names_in_'):
            self.feature_names = model.feature_names_in_.tolist()
        else:
            # Create generic feature names
            n_features = getattr(model, 'n_features_in_', 5)
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
    
    def extract_single_tree_structure(self, tree_id: int) -> Dict[str, Any]:
        """
        Extract the complete structure of a single tree
        
        This function implements T2.1.1 - Create function to extract single tree structure
        
        Args:
            tree_id (int): Index of the tree to extract (0 to n_estimators-1)
            
        Returns:
            Dict[str, Any]: Complete tree structure with all nodes and relationships
        """
        if tree_id < 0 or tree_id >= self.model.n_estimators:
            logger.error(f"Invalid tree ID: {tree_id}. Must be between 0 and {self.model.n_estimators-1}")
            raise ValueError(f"Invalid tree ID: {tree_id}. Must be between 0 and {self.model.n_estimators-1}")
        
        try:
            logger.info(f"Extracting structure for tree {tree_id}")
            
            # Get the specific tree
            tree_estimator = self.model.estimators_[tree_id]
            tree_structure = tree_estimator.tree_
            
            # Extract basic tree information
            tree_info = {
                "tree_id": tree_id,
                "metadata": {
                    "total_nodes": tree_structure.node_count,
                    "max_depth": tree_structure.max_depth,
                    "n_features": tree_structure.n_features,
                    "n_outputs": tree_structure.n_outputs,
                    "n_classes": len(tree_structure.value[0][0]) if tree_structure.value.shape[2] > 1 else 1
                },
                "nodes": [],
                "structure_info": {
                    "leaf_count": 0,
                    "internal_node_count": 0,
                    "feature_usage": {},
                    "depth_distribution": {}
                }
            }
            
            # Extract all nodes
            nodes_data = self._extract_all_nodes(tree_structure)
            tree_info["nodes"] = nodes_data["nodes"]
            tree_info["structure_info"] = nodes_data["structure_info"]
            
            logger.info(f"Successfully extracted tree {tree_id}: {tree_info['metadata']['total_nodes']} nodes, depth {tree_info['metadata']['max_depth']}")
            return tree_info
            
        except Exception as e:
            logger.error(f"Error extracting tree {tree_id} structure: {str(e)}")
            raise Exception(f"Failed to extract tree {tree_id}: {str(e)}")
    
    def _extract_all_nodes(self, tree_structure) -> Dict[str, Any]:
        """
        Extract all nodes from a tree structure with complete information
        
        Args:
            tree_structure: sklearn tree structure
            
        Returns:
            Dict[str, Any]: All nodes with structure information
        """
        nodes = []
        structure_info = {
            "leaf_count": 0,
            "internal_node_count": 0,
            "feature_usage": {},
            "depth_distribution": {}
        }
        
        # Calculate node depths
        node_depths = self._calculate_node_depths(tree_structure)
        
        # Extract each node
        for node_id in range(tree_structure.node_count):
            node_info = self._extract_single_node(tree_structure, node_id, node_depths[node_id])
            nodes.append(node_info)
            
            # Update structure statistics
            if node_info["is_leaf"]:
                structure_info["leaf_count"] += 1
            else:
                structure_info["internal_node_count"] += 1
                
                # Track feature usage
                feature_name = node_info["feature_name"]
                structure_info["feature_usage"][feature_name] = structure_info["feature_usage"].get(feature_name, 0) + 1
            
            # Track depth distribution
            depth = node_info["depth"]
            structure_info["depth_distribution"][str(depth)] = structure_info["depth_distribution"].get(str(depth), 0) + 1
        
        return {
            "nodes": nodes,
            "structure_info": structure_info
        }
    
    def _extract_single_node(self, tree_structure, node_id: int, depth: int) -> Dict[str, Any]:
        """
        Extract detailed information for a single node
        
        Args:
            tree_structure: sklearn tree structure
            node_id (int): ID of the node to extract
            depth (int): Depth of the node in the tree
            
        Returns:
            Dict[str, Any]: Complete node information
        """
        # Check if this is a leaf node
        is_leaf = tree_structure.children_left[node_id] == -1
        
        # Basic node information
        node_info = {
            "node_id": node_id,
            "depth": depth,
            "is_leaf": is_leaf,
            "samples": int(tree_structure.n_node_samples[node_id]),
            "impurity": float(tree_structure.impurity[node_id]),
            "value": tree_structure.value[node_id].tolist(),
        }
        
        if is_leaf:
            # Leaf node specific information
            node_info.update({
                "feature_name": None,
                "feature_index": -1,
                "threshold": None,
                "left_child": None,
                "right_child": None,
                "prediction": self._get_node_prediction(tree_structure.value[node_id]),
                "class_distribution": self._get_class_distribution(tree_structure.value[node_id])
            })
        else:
            # Internal node specific information
            feature_index = tree_structure.feature[node_id]
            feature_name = self.feature_names[feature_index] if feature_index < len(self.feature_names) else f"feature_{feature_index}"
            
            node_info.update({
                "feature_name": feature_name,
                "feature_index": int(feature_index),
                "threshold": float(tree_structure.threshold[node_id]),
                "left_child": int(tree_structure.children_left[node_id]),
                "right_child": int(tree_structure.children_right[node_id]),
                "prediction": self._get_node_prediction(tree_structure.value[node_id]),
                "class_distribution": self._get_class_distribution(tree_structure.value[node_id])
            })
        
        return node_info
    
    def _calculate_node_depths(self, tree_structure) -> List[int]:
        """
        Calculate the depth of each node in the tree
        
        Args:
            tree_structure: sklearn tree structure
            
        Returns:
            List[int]: Depth of each node
        """
        depths = [-1] * tree_structure.node_count
        depths[0] = 0  # Root node has depth 0
        
        # Use BFS to calculate depths
        queue = [0]  # Start with root node
        
        while queue:
            current_node = queue.pop(0)
            current_depth = depths[current_node]
            
            # Process left child
            left_child = tree_structure.children_left[current_node]
            if left_child != -1:
                depths[left_child] = current_depth + 1
                queue.append(left_child)
            
            # Process right child
            right_child = tree_structure.children_right[current_node]
            if right_child != -1:
                depths[right_child] = current_depth + 1
                queue.append(right_child)
        
        return depths
    
    def _get_node_prediction(self, node_value) -> Union[float, int]:
        """
        Get the prediction for a node based on its value
        
        Args:
            node_value: Node value array from sklearn tree
            
        Returns:
            Union[float, int]: Prediction value
        """
        if len(node_value.shape) == 3 and node_value.shape[2] > 1:
            # Classification: return class with highest count
            class_counts = node_value[0][0]
            return int(np.argmax(class_counts))
        else:
            # Regression or binary classification: return the value
            value = node_value[0][0][0] if len(node_value.shape) == 3 else node_value[0][0]
            return float(value)
    
    def _get_class_distribution(self, node_value) -> Dict[str, Any]:
        """
        Get class distribution for a node
        
        Args:
            node_value: Node value array from sklearn tree
            
        Returns:
            Dict[str, Any]: Class distribution information
        """
        if len(node_value.shape) == 3 and node_value.shape[2] > 1:
            # Multi-class classification
            class_counts = node_value[0][0]
            total_samples = np.sum(class_counts)
            
            distribution = {
                "class_counts": class_counts.tolist(),
                "class_probabilities": (class_counts / total_samples).tolist() if total_samples > 0 else [0] * len(class_counts),
                "total_samples": int(total_samples),
                "predicted_class": int(np.argmax(class_counts)),
                "confidence": float(np.max(class_counts) / total_samples) if total_samples > 0 else 0.0
            }
        else:
            # Binary classification or regression
            value = node_value[0][0][0] if len(node_value.shape) == 3 else node_value[0][0]
            distribution = {
                "value": float(value),
                "total_samples": int(node_value[0][0][0]) if len(node_value.shape) == 3 else 1,
                "confidence": 1.0
            }
        
        return distribution
    
    def convert_tree_to_json(self, tree_id: int) -> Dict[str, Any]:
        """
        Convert sklearn tree to JSON format with complete structure
        
        This function implements T2.1.2 - Convert sklearn tree to JSON format
        
        Args:
            tree_id (int): Index of the tree to convert
            
        Returns:
            Dict[str, Any]: Tree structure in JSON-serializable format
        """
        try:
            logger.info(f"Converting tree {tree_id} to JSON format")
            
            # Extract the tree structure
            tree_structure = self.extract_single_tree_structure(tree_id)
            
            # Create JSON-friendly format
            json_tree = {
                "tree_id": tree_id,
                "format_version": "1.0",
                "extraction_timestamp": None,  # Will be set by caller if needed
                "metadata": tree_structure["metadata"],
                "root_node": self._build_hierarchical_structure(tree_structure["nodes"]),
                "flat_nodes": tree_structure["nodes"],
                "statistics": tree_structure["structure_info"],
                "feature_names": self.feature_names,
                "traversal_info": {
                    "total_paths": 2 ** tree_structure["structure_info"]["leaf_count"] if tree_structure["structure_info"]["leaf_count"] > 0 else 1,
                    "max_depth": tree_structure["metadata"]["max_depth"],
                    "branching_factor": 2  # Binary tree
                }
            }
            
            # Ensure JSON serializable
            json_tree = self._make_json_serializable(json_tree)
            
            logger.info(f"Successfully converted tree {tree_id} to JSON format")
            return json_tree
            
        except Exception as e:
            logger.error(f"Error converting tree {tree_id} to JSON: {str(e)}")
            raise Exception(f"Failed to convert tree {tree_id} to JSON: {str(e)}")
    
    def _build_hierarchical_structure(self, flat_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build hierarchical tree structure from flat node list
        
        Args:
            flat_nodes (List[Dict[str, Any]]): Flat list of all nodes
            
        Returns:
            Dict[str, Any]: Hierarchical tree structure starting from root
        """
        # Create lookup dictionary for quick access
        node_lookup = {node["node_id"]: node.copy() for node in flat_nodes}
        
        # Build hierarchical structure
        for node in node_lookup.values():
            if not node["is_leaf"]:
                # Add children to internal nodes
                left_child_id = node["left_child"]
                right_child_id = node["right_child"]
                
                node["children"] = {
                    "left": node_lookup[left_child_id] if left_child_id is not None else None,
                    "right": node_lookup[right_child_id] if right_child_id is not None else None
                }
            else:
                node["children"] = None
        
        # Return root node (node_id = 0)
        return node_lookup[0]
    
    def get_tree_node_info(self, tree_id: int, include_children: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive node information for a tree
        
        This function implements T2.1.3 - Include node information (feature, threshold, samples, gini)
        
        Args:
            tree_id (int): Index of the tree
            include_children (bool): Whether to include children node references
            
        Returns:
            Dict[str, Any]: Comprehensive node information
        """
        try:
            logger.info(f"Getting comprehensive node info for tree {tree_id}")
            
            tree_structure = self.extract_single_tree_structure(tree_id)
            
            # Enhance nodes with additional information
            enhanced_nodes = []
            for node in tree_structure["nodes"]:
                enhanced_node = node.copy()
                
                # Add additional node metrics
                enhanced_node["node_metrics"] = {
                    "gini_impurity": node["impurity"],
                    "sample_ratio": node["samples"] / tree_structure["nodes"][0]["samples"] if tree_structure["nodes"][0]["samples"] > 0 else 0,
                    "depth_ratio": node["depth"] / tree_structure["metadata"]["max_depth"] if tree_structure["metadata"]["max_depth"] > 0 else 0,
                    "is_pure": node["impurity"] < 1e-7,  # Essentially pure node
                    "information_gain": None  # Will be calculated if needed
                }
                
                # Add decision information for internal nodes
                if not node["is_leaf"]:
                    enhanced_node["decision_info"] = {
                        "split_condition": f"{node['feature_name']} <= {node['threshold']:.6f}",
                        "feature_type": "continuous",  # Assuming continuous features
                        "split_quality": 1.0 - node["impurity"]  # Higher is better
                    }
                
                # Add children references if requested
                if include_children and not node["is_leaf"]:
                    enhanced_node["children_references"] = {
                        "left_child_id": node["left_child"],
                        "right_child_id": node["right_child"],
                        "has_left_child": node["left_child"] is not None,
                        "has_right_child": node["right_child"] is not None
                    }
                
                enhanced_nodes.append(enhanced_node)
            
            result = {
                "tree_id": tree_id,
                "node_count": len(enhanced_nodes),
                "nodes": enhanced_nodes,
                "tree_metadata": tree_structure["metadata"],
                "structure_summary": tree_structure["structure_info"]
            }
            
            # Ensure JSON serializable
            result = self._make_json_serializable(result)
            
            logger.info(f"Successfully generated comprehensive node info for tree {tree_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting node info for tree {tree_id}: {str(e)}")
            raise Exception(f"Failed to get node info for tree {tree_id}: {str(e)}")
    
    def add_children_node_references(self, tree_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add children node references to tree structure
        
        This function implements T2.1.4 - Add children node references
        
        Args:
            tree_structure (Dict[str, Any]): Tree structure to enhance
            
        Returns:
            Dict[str, Any]: Enhanced tree structure with children references
        """
        try:
            logger.info(f"Adding children node references to tree {tree_structure.get('tree_id', 'unknown')}")
            
            enhanced_structure = tree_structure.copy()
            enhanced_nodes = []
            
            # Create node lookup for quick access
            node_lookup = {node["node_id"]: node for node in tree_structure["nodes"]}
            
            for node in tree_structure["nodes"]:
                enhanced_node = node.copy()
                
                if not node["is_leaf"]:
                    # Add detailed children information
                    left_child_id = node["left_child"]
                    right_child_id = node["right_child"]
                    
                    enhanced_node["children_details"] = {
                        "left_child": {
                            "node_id": left_child_id,
                            "is_leaf": node_lookup[left_child_id]["is_leaf"] if left_child_id in node_lookup else False,
                            "samples": node_lookup[left_child_id]["samples"] if left_child_id in node_lookup else 0,
                            "impurity": node_lookup[left_child_id]["impurity"] if left_child_id in node_lookup else 0,
                            "prediction": node_lookup[left_child_id]["prediction"] if left_child_id in node_lookup else None
                        },
                        "right_child": {
                            "node_id": right_child_id,
                            "is_leaf": node_lookup[right_child_id]["is_leaf"] if right_child_id in node_lookup else False,
                            "samples": node_lookup[right_child_id]["samples"] if right_child_id in node_lookup else 0,
                            "impurity": node_lookup[right_child_id]["impurity"] if right_child_id in node_lookup else 0,
                            "prediction": node_lookup[right_child_id]["prediction"] if right_child_id in node_lookup else None
                        }
                    }
                    
                    # Add navigation helpers
                    enhanced_node["navigation"] = {
                        "has_children": True,
                        "child_count": 2,
                        "left_path": f"node_{left_child_id}",
                        "right_path": f"node_{right_child_id}",
                        "subtree_size": self._calculate_subtree_size(node_lookup, node["node_id"])
                    }
                else:
                    enhanced_node["children_details"] = None
                    enhanced_node["navigation"] = {
                        "has_children": False,
                        "child_count": 0,
                        "is_terminal": True,
                        "subtree_size": 1
                    }
                
                enhanced_nodes.append(enhanced_node)
            
            enhanced_structure["nodes"] = enhanced_nodes
            
            # Add tree navigation summary
            enhanced_structure["navigation_summary"] = {
                "total_nodes": len(enhanced_nodes),
                "internal_nodes": len([n for n in enhanced_nodes if not n["is_leaf"]]),
                "leaf_nodes": len([n for n in enhanced_nodes if n["is_leaf"]]),
                "max_depth": max(n["depth"] for n in enhanced_nodes),
                "root_node_id": 0,
                "traversal_paths": self._generate_traversal_paths(enhanced_nodes)
            }
            
            logger.info(f"Successfully added children references to tree {tree_structure.get('tree_id', 'unknown')}")
            return enhanced_structure
            
        except Exception as e:
            logger.error(f"Error adding children references: {str(e)}")
            raise Exception(f"Failed to add children references: {str(e)}")
    
    def _calculate_subtree_size(self, node_lookup: Dict[int, Dict[str, Any]], node_id: int) -> int:
        """
        Calculate the size of subtree rooted at given node
        
        Args:
            node_lookup (Dict[int, Dict[str, Any]]): Lookup dictionary for nodes
            node_id (int): Root node of subtree
            
        Returns:
            int: Number of nodes in subtree
        """
        if node_id not in node_lookup:
            return 0
        
        node = node_lookup[node_id]
        if node["is_leaf"]:
            return 1
        
        left_size = self._calculate_subtree_size(node_lookup, node["left_child"]) if node["left_child"] is not None else 0
        right_size = self._calculate_subtree_size(node_lookup, node["right_child"]) if node["right_child"] is not None else 0
        
        return 1 + left_size + right_size
    
    def _generate_traversal_paths(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate information about possible traversal paths in the tree
        
        Args:
            nodes (List[Dict[str, Any]]): List of all nodes
            
        Returns:
            Dict[str, Any]: Traversal path information
        """
        leaf_nodes = [n for n in nodes if n["is_leaf"]]
        internal_nodes = [n for n in nodes if not n["is_leaf"]]
        
        return {
            "total_paths_to_leaves": len(leaf_nodes),
            "average_path_length": sum(n["depth"] for n in leaf_nodes) / len(leaf_nodes) if leaf_nodes else 0,
            "shortest_path": min(n["depth"] for n in leaf_nodes) if leaf_nodes else 0,
            "longest_path": max(n["depth"] for n in leaf_nodes) if leaf_nodes else 0,
            "decision_points": len(internal_nodes),
            "leaf_distribution_by_depth": self._get_leaf_distribution_by_depth(leaf_nodes)
        }
    
    def _get_leaf_distribution_by_depth(self, leaf_nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get distribution of leaf nodes by depth
        
        Args:
            leaf_nodes (List[Dict[str, Any]]): List of leaf nodes
            
        Returns:
            Dict[str, int]: Distribution of leaves by depth
        """
        distribution = {}
        for leaf in leaf_nodes:
            depth = str(leaf["depth"])
            distribution[depth] = distribution.get(depth, 0) + 1
        return distribution
    
    def convert_tree_to_json_with_depth_limit(self, tree_id: int, max_depth: Optional[int] = None) -> Dict[str, Any]:
        """
        Convert sklearn tree to JSON format with depth limitation
        
        Args:
            tree_id (int): Index of the tree to convert
            max_depth (Optional[int]): Maximum depth to include (None for no limit)
            
        Returns:
            Dict[str, Any]: Tree structure in JSON-serializable format with depth limit
        """
        try:
            logger.info(f"Converting tree {tree_id} to JSON format with max_depth={max_depth}")
            
            # Extract the full tree structure first
            tree_structure = self.extract_single_tree_structure(tree_id)
            
            # Filter nodes by depth if max_depth is specified
            if max_depth is not None:
                filtered_nodes = [node for node in tree_structure["nodes"] if node["depth"] <= max_depth]
                
                # Update structure info for filtered nodes
                filtered_structure_info = {
                    "leaf_count": 0,
                    "internal_node_count": 0,
                    "feature_usage": {},
                    "depth_distribution": {},
                    "truncated_nodes": 0,
                    "original_total_nodes": len(tree_structure["nodes"]),
                    "filtered_total_nodes": len(filtered_nodes)
                }
                
                # Recalculate statistics for filtered nodes
                for node in filtered_nodes:
                    if node["is_leaf"] or node["depth"] == max_depth:
                        # Nodes at max_depth are treated as leaves even if they have children
                        filtered_structure_info["leaf_count"] += 1
                        if not node["is_leaf"] and node["depth"] == max_depth:
                            filtered_structure_info["truncated_nodes"] += 1
                    else:
                        filtered_structure_info["internal_node_count"] += 1
                        
                        # Track feature usage
                        feature_name = node["feature_name"]
                        filtered_structure_info["feature_usage"][feature_name] = filtered_structure_info["feature_usage"].get(feature_name, 0) + 1
                    
                    # Track depth distribution
                    depth = node["depth"]
                    filtered_structure_info["depth_distribution"][str(depth)] = filtered_structure_info["depth_distribution"].get(str(depth), 0) + 1
                
                # Update nodes to mark truncated ones
                for node in filtered_nodes:
                    if not node["is_leaf"] and node["depth"] == max_depth:
                        # Mark this node as truncated
                        node["is_truncated"] = True
                        node["original_is_leaf"] = False
                        node["truncated_children_count"] = self._count_children_beyond_depth(tree_structure["nodes"], node["node_id"])
                    else:
                        node["is_truncated"] = False
                        node["original_is_leaf"] = node["is_leaf"]
                        node["truncated_children_count"] = 0
                
                tree_structure["nodes"] = filtered_nodes
                tree_structure["structure_info"] = filtered_structure_info
                
                # Update metadata
                tree_structure["metadata"]["effective_max_depth"] = min(max_depth, tree_structure["metadata"]["max_depth"])
                tree_structure["metadata"]["is_depth_limited"] = True
                tree_structure["metadata"]["depth_limit"] = max_depth
            else:
                # Add metadata indicating no depth limit
                tree_structure["metadata"]["effective_max_depth"] = tree_structure["metadata"]["max_depth"]
                tree_structure["metadata"]["is_depth_limited"] = False
                tree_structure["metadata"]["depth_limit"] = None
                
                # Mark all nodes as not truncated
                for node in tree_structure["nodes"]:
                    node["is_truncated"] = False
                    node["original_is_leaf"] = node["is_leaf"]
                    node["truncated_children_count"] = 0
            
            # Create JSON-friendly format
            json_tree = {
                "tree_id": tree_id,
                "format_version": "1.1",  # Updated version to indicate depth support
                "extraction_timestamp": None,  # Will be set by caller if needed
                "metadata": tree_structure["metadata"],
                "root_node": self._build_hierarchical_structure_with_depth_limit(tree_structure["nodes"], max_depth),
                "flat_nodes": tree_structure["nodes"],
                "statistics": tree_structure["structure_info"],
                "feature_names": self.feature_names,
                "traversal_info": {
                    "total_paths": 2 ** tree_structure["structure_info"]["leaf_count"] if tree_structure["structure_info"]["leaf_count"] > 0 else 1,
                    "max_depth": tree_structure["metadata"]["effective_max_depth"],
                    "branching_factor": 2,  # Binary tree
                    "depth_limited": tree_structure["metadata"]["is_depth_limited"],
                    "truncated_paths": tree_structure["structure_info"].get("truncated_nodes", 0)
                }
            }
            
            # Ensure JSON serializable
            json_tree = self._make_json_serializable(json_tree)
            
            logger.info(f"Successfully converted tree {tree_id} to JSON format with depth limit {max_depth}")
            return json_tree
            
        except Exception as e:
            logger.error(f"Error converting tree {tree_id} to JSON with depth limit: {str(e)}")
            raise Exception(f"Failed to convert tree {tree_id} to JSON with depth limit: {str(e)}")
    
    def _count_children_beyond_depth(self, all_nodes: List[Dict[str, Any]], parent_node_id: int) -> int:
        """
        Count how many children exist beyond the current depth limit for a given node
        
        Args:
            all_nodes (List[Dict[str, Any]]): All nodes in the tree
            parent_node_id (int): ID of the parent node
            
        Returns:
            int: Number of children beyond the depth limit
        """
        node_lookup = {node["node_id"]: node for node in all_nodes}
        parent_node = node_lookup.get(parent_node_id)
        
        if not parent_node or parent_node["is_leaf"]:
            return 0
        
        # Count all descendants
        def count_descendants(node_id: int) -> int:
            node = node_lookup.get(node_id)
            if not node or node["is_leaf"]:
                return 1
            
            left_count = count_descendants(node["left_child"]) if node["left_child"] is not None else 0
            right_count = count_descendants(node["right_child"]) if node["right_child"] is not None else 0
            
            return 1 + left_count + right_count
        
        total_descendants = 0
        if parent_node["left_child"] is not None:
            total_descendants += count_descendants(parent_node["left_child"])
        if parent_node["right_child"] is not None:
            total_descendants += count_descendants(parent_node["right_child"])
        
        return total_descendants
    
    def _build_hierarchical_structure_with_depth_limit(self, flat_nodes: List[Dict[str, Any]], max_depth: Optional[int] = None) -> Dict[str, Any]:
        """
        Build hierarchical tree structure from flat node list with depth limitation
        
        Args:
            flat_nodes (List[Dict[str, Any]]): Flat list of nodes (already filtered by depth)
            max_depth (Optional[int]): Maximum depth limit
            
        Returns:
            Dict[str, Any]: Hierarchical tree structure starting from root
        """
        # Create lookup dictionary for quick access
        node_lookup = {node["node_id"]: node.copy() for node in flat_nodes}
        
        # Build hierarchical structure
        for node in node_lookup.values():
            if not node["is_leaf"] and (max_depth is None or node["depth"] < max_depth):
                # Add children to internal nodes that are not at the depth limit
                left_child_id = node["left_child"]
                right_child_id = node["right_child"]
                
                node["children"] = {
                    "left": node_lookup.get(left_child_id) if left_child_id in node_lookup else None,
                    "right": node_lookup.get(right_child_id) if right_child_id in node_lookup else None
                }
            else:
                node["children"] = None
        
        # Return root node (node_id = 0)
        return node_lookup[0]

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types and other non-JSON serializable types to JSON serializable types
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def validate_tree_id(self, tree_id: int) -> bool:
        """
        Validate if tree ID is valid for the model
        
        This function implements T2.1.7 - Add error handling for invalid tree IDs
        
        Args:
            tree_id (int): Tree ID to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return 0 <= tree_id < self.model.n_estimators
    
    def get_tree_extraction_summary(self) -> Dict[str, Any]:
        """
        Get summary of tree extraction capabilities
        
        Returns:
            Dict[str, Any]: Summary of extraction capabilities
        """
        return {
            "model_info": {
                "total_trees": self.model.n_estimators,
                "feature_count": len(self.feature_names),
                "feature_names": self.feature_names[:10],  # First 10 features
                "model_type": type(self.model).__name__
            },
            "extraction_capabilities": {
                "single_tree_extraction": True,
                "json_conversion": True,
                "node_information": True,
                "children_references": True,
                "hierarchical_structure": True,
                "navigation_helpers": True
            },
            "supported_formats": [
                "flat_node_list",
                "hierarchical_tree",
                "json_serializable",
                "enhanced_with_metrics"
            ],
            "validation": {
                "tree_id_validation": True,
                "error_handling": True,
                "comprehensive_logging": True
            }
        }

# Convenience functions for easy access
def extract_tree_structure(model, tree_id: int) -> Dict[str, Any]:
    """
    Convenience function to extract tree structure
    
    Args:
        model: Random Forest model
        tree_id (int): Tree ID to extract
        
    Returns:
        Dict[str, Any]: Tree structure
    """
    extractor = TreeStructureExtractor(model)
    return extractor.extract_single_tree_structure(tree_id)

def convert_tree_to_json_format(model, tree_id: int) -> Dict[str, Any]:
    """
    Convenience function to convert tree to JSON
    
    Args:
        model: Random Forest model
        tree_id (int): Tree ID to convert
        
    Returns:
        Dict[str, Any]: JSON tree structure
    """
    extractor = TreeStructureExtractor(model)
    return extractor.convert_tree_to_json(tree_id)

def get_comprehensive_tree_info(model, tree_id: int) -> Dict[str, Any]:
    """
    Convenience function to get comprehensive tree information
    
    Args:
        model: Random Forest model
        tree_id (int): Tree ID to analyze
        
    Returns:
        Dict[str, Any]: Comprehensive tree information
    """
    extractor = TreeStructureExtractor(model)
    return extractor.get_tree_node_info(tree_id, include_children=True)

if __name__ == "__main__":
    # Test the tree extractor
    try:
        from model_loader import load_random_forest_model
        
        print("Testing Tree Structure Extractor...")
        
        # Load model
        model_path = "../data/random_forest_model.pkl"
        loader = load_random_forest_model(model_path)
        
        # Create extractor
        extractor = TreeStructureExtractor(loader.model)
        
        # Test extraction
        tree_structure = extractor.extract_single_tree_structure(0)
        print(f"Extracted tree 0: {tree_structure['metadata']['total_nodes']} nodes")
        
        # Test JSON conversion
        json_tree = extractor.convert_tree_to_json(0)
        print(f"Converted to JSON: {len(json_tree['flat_nodes'])} nodes")
        
        # Test comprehensive info
        comprehensive_info = extractor.get_tree_node_info(0)
        print(f"Comprehensive info: {comprehensive_info['node_count']} nodes with enhanced data")
        
        print("Tree Structure Extractor test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
