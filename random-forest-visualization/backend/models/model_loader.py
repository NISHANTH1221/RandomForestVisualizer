"""
Model Loader Module

This module handles loading and analyzing the Random Forest model from the pickle file.
It provides functions to:
- Load the trained Random Forest model
- Extract basic model information (n_estimators, features, etc.)
- Test model predictions with sample data
- Verify model structure and tree access
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class RandomForestModelLoader:
    """
    A class to handle loading and analyzing the Random Forest model
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the model loader with the path to the pickle file
        
        Args:
            model_path (str): Path to the pickle file containing the Random Forest model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.model_info = {}
        
    def load_model(self) -> bool:
        """
        Load the Random Forest model from the pickle file
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
                
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def extract_model_info(self) -> Dict[str, Any]:
        """
        Extract basic information about the Random Forest model
        
        Returns:
            Dict[str, Any]: Dictionary containing model information
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        try:
            # Extract basic model information
            self.model_info = {
                "model_type": type(self.model).__name__,
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "min_samples_split": self.model.min_samples_split,
                "min_samples_leaf": self.model.min_samples_leaf,
                "max_features": self.model.max_features,
                "bootstrap": self.model.bootstrap,
                "random_state": self.model.random_state,
                "n_features_in": getattr(self.model, 'n_features_in_', None),
                "feature_names_in": getattr(self.model, 'feature_names_in_', None),
                "n_classes": getattr(self.model, 'n_classes_', None),
                "classes": getattr(self.model, 'classes_', None).tolist() if hasattr(self.model, 'classes_') else None
            }
            
            # Store feature names for later use
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
            else:
                # If feature names not available, create generic names
                n_features = getattr(self.model, 'n_features_in_', 5)
                self.feature_names = [f"feature_{i}" for i in range(n_features)]
            
            self.model_info["feature_names"] = self.feature_names
            
            logger.info(f"Extracted model info: {self.model_info['n_estimators']} trees, {len(self.feature_names)} features")
            return self.model_info
            
        except Exception as e:
            logger.error(f"Error extracting model info: {str(e)}")
            return {}
    
    def get_tree_info(self, tree_index: int) -> Dict[str, Any]:
        """
        Get information about a specific tree in the Random Forest
        
        Args:
            tree_index (int): Index of the tree (0 to n_estimators-1)
            
        Returns:
            Dict[str, Any]: Dictionary containing tree information
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        if tree_index < 0 or tree_index >= self.model.n_estimators:
            logger.error(f"Invalid tree index: {tree_index}. Must be between 0 and {self.model.n_estimators-1}")
            return {}
            
        try:
            tree = self.model.estimators_[tree_index]
            tree_structure = tree.tree_
            
            tree_info = {
                "tree_index": tree_index,
                "node_count": tree_structure.node_count,
                "max_depth": tree_structure.max_depth,
                "n_features": tree_structure.n_features,
                "n_outputs": tree_structure.n_outputs,
                "feature_importances": tree.feature_importances_.tolist() if hasattr(tree, 'feature_importances_') else None
            }
            
            return tree_info
            
        except Exception as e:
            logger.error(f"Error getting tree info for tree {tree_index}: {str(e)}")
            return {}
    
    def test_prediction(self, sample_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test the model with sample data to verify it's working correctly
        
        Args:
            sample_data (Optional[Dict[str, Any]]): Sample input data. If None, creates default sample.
            
        Returns:
            Dict[str, Any]: Dictionary containing prediction results
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        try:
            # Create sample data if not provided
            if sample_data is None:
                # Create a default sample based on expected features
                # This should match the features used in training
                sample_data = {
                    "error_message": "insufficient_funds",
                    "billing_state": "CA", 
                    "card_funding": "credit",
                    "card_network": "visa",
                    "card_issuer": "chase"
                }
            
            # Convert sample data to the format expected by the model
            # Note: This will need to be updated based on the actual feature encoding
            sample_array = self._prepare_sample_data(sample_data)
            
            # Make prediction
            prediction = self.model.predict(sample_array)
            prediction_proba = self.model.predict_proba(sample_array)
            
            # Get individual tree predictions
            individual_predictions = []
            for i, tree in enumerate(self.model.estimators_):
                tree_pred = tree.predict(sample_array)[0]
                tree_proba = tree.predict_proba(sample_array)[0] if hasattr(tree, 'predict_proba') else None
                individual_predictions.append({
                    "tree_id": i,
                    "prediction": float(tree_pred),
                    "probability": tree_proba.tolist() if tree_proba is not None else None
                })
            
            result = {
                "sample_input": sample_data,
                "ensemble_prediction": float(prediction[0]),
                "ensemble_probability": prediction_proba[0].tolist() if prediction_proba is not None else None,
                "individual_predictions": individual_predictions[:5],  # Show first 5 trees as example
                "total_trees": len(individual_predictions)
            }
            
            logger.info(f"Test prediction successful. Ensemble prediction: {result['ensemble_prediction']}")
            return result
            
        except Exception as e:
            logger.error(f"Error during test prediction: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_sample_data(self, sample_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare sample data for model prediction using proper one-hot encoding
        
        Args:
            sample_data (Dict[str, Any]): Raw sample data
            
        Returns:
            np.ndarray: Prepared data array for model input
        """
        try:
            # Load parameter encoding
            import json
            from pathlib import Path
            
            encoding_path = Path(__file__).parent.parent / "data" / "param_encoding.json"
            with open(encoding_path, 'r') as f:
                param_encoding = json.load(f)
            
            # Create one-hot encoded features
            encoded_features = []
            
            # Map sample_data keys to encoding keys
            key_mapping = {
                "error_message": "first_error_message",
                "billing_state": "billing_state", 
                "card_funding": "card_funding",
                "card_network": "card_network",
                "card_issuer": "card_issuer"
            }
            
            for sample_key, encoding_key in key_mapping.items():
                if sample_key in sample_data and encoding_key in param_encoding:
                    value = sample_data[sample_key]
                    categories = param_encoding[encoding_key]
                    
                    # Create one-hot encoding
                    one_hot = np.zeros(len(categories))
                    if value in categories:
                        one_hot[categories.index(value)] = 1
                    
                    encoded_features.extend(one_hot)
                else:
                    # If key not found, add zeros for that category
                    if encoding_key in param_encoding:
                        encoded_features.extend(np.zeros(len(param_encoding[encoding_key])))
            
            # Add placeholder values for time-based features
            # weekday, day, hour (3 features total to make 114)
            time_features = [1, 15, 10]  # Monday, 15th day, 10 AM
            encoded_features.extend(time_features)
            
            return np.array([encoded_features], dtype=float)
            
        except Exception as e:
            logger.error(f"Error preparing sample data: {str(e)}")
            # Fallback to zeros if encoding fails
            return np.zeros((1, 114), dtype=float)
    
    def extract_tree_count_and_stats(self) -> Dict[str, Any]:
        """
        Extract tree count and basic statistics for all trees in the Random Forest
        
        This function implements T1.4.1 - Build function to extract tree count and basic stats
        
        Returns:
            Dict[str, Any]: Dictionary containing tree count and comprehensive statistics
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        try:
            logger.info("Extracting tree count and basic statistics...")
            
            # Initialize statistics containers
            tree_stats = {
                "total_trees": self.model.n_estimators,
                "trees_analyzed": 0,
                "depth_stats": {
                    "min_depth": float('inf'),
                    "max_depth": 0,
                    "avg_depth": 0,
                    "depth_distribution": {}
                },
                "node_stats": {
                    "min_nodes": float('inf'),
                    "max_nodes": 0,
                    "avg_nodes": 0,
                    "total_nodes": 0,
                    "node_distribution": {}
                },
                "feature_usage": {},
                "tree_summaries": [],
                "model_health": {
                    "trees_accessible": 0,
                    "trees_with_errors": 0,
                    "error_details": []
                }
            }
            
            depths = []
            node_counts = []
            
            # Analyze each tree
            for tree_idx in range(self.model.n_estimators):
                try:
                    tree = self.model.estimators_[tree_idx]
                    tree_structure = tree.tree_
                    
                    # Extract basic tree information
                    tree_depth = tree_structure.max_depth
                    tree_nodes = tree_structure.node_count
                    
                    # Update statistics
                    depths.append(tree_depth)
                    node_counts.append(tree_nodes)
                    
                    # Update min/max values
                    tree_stats["depth_stats"]["min_depth"] = min(tree_stats["depth_stats"]["min_depth"], tree_depth)
                    tree_stats["depth_stats"]["max_depth"] = max(tree_stats["depth_stats"]["max_depth"], tree_depth)
                    tree_stats["node_stats"]["min_nodes"] = min(tree_stats["node_stats"]["min_nodes"], tree_nodes)
                    tree_stats["node_stats"]["max_nodes"] = max(tree_stats["node_stats"]["max_nodes"], tree_nodes)
                    tree_stats["node_stats"]["total_nodes"] += tree_nodes
                    
                    # Track depth distribution
                    depth_key = str(tree_depth)
                    tree_stats["depth_stats"]["depth_distribution"][depth_key] = \
                        tree_stats["depth_stats"]["depth_distribution"].get(depth_key, 0) + 1
                    
                    # Track node count distribution (grouped by ranges)
                    node_range = self._get_node_range(tree_nodes)
                    tree_stats["node_stats"]["node_distribution"][node_range] = \
                        tree_stats["node_stats"]["node_distribution"].get(node_range, 0) + 1
                    
                    # Extract feature usage for this tree
                    features_used = tree_structure.feature[tree_structure.feature >= 0]
                    for feature_idx in features_used:
                        feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                        tree_stats["feature_usage"][feature_name] = \
                            tree_stats["feature_usage"].get(feature_name, 0) + 1
                    
                    # Create tree summary
                    tree_summary = {
                        "tree_id": tree_idx,
                        "depth": tree_depth,
                        "nodes": tree_nodes,
                        "features_used": len(set(features_used)),
                        "leaf_nodes": np.sum(tree_structure.children_left == -1),
                        "internal_nodes": tree_nodes - np.sum(tree_structure.children_left == -1)
                    }
                    tree_stats["tree_summaries"].append(tree_summary)
                    
                    tree_stats["model_health"]["trees_accessible"] += 1
                    tree_stats["trees_analyzed"] += 1
                    
                except Exception as tree_error:
                    logger.warning(f"Error analyzing tree {tree_idx}: {str(tree_error)}")
                    tree_stats["model_health"]["trees_with_errors"] += 1
                    tree_stats["model_health"]["error_details"].append({
                        "tree_id": tree_idx,
                        "error": str(tree_error)
                    })
            
            # Calculate averages
            if depths:
                tree_stats["depth_stats"]["avg_depth"] = np.mean(depths)
            if node_counts:
                tree_stats["node_stats"]["avg_nodes"] = np.mean(node_counts)
            
            # Sort feature usage by frequency
            tree_stats["feature_usage"] = dict(sorted(
                tree_stats["feature_usage"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # Add summary statistics
            tree_stats["summary"] = {
                "total_trees": tree_stats["total_trees"],
                "successfully_analyzed": tree_stats["trees_analyzed"],
                "analysis_success_rate": tree_stats["trees_analyzed"] / tree_stats["total_trees"] * 100,
                "avg_depth": round(tree_stats["depth_stats"]["avg_depth"], 2),
                "avg_nodes": round(tree_stats["node_stats"]["avg_nodes"], 2),
                "total_nodes_all_trees": tree_stats["node_stats"]["total_nodes"],
                "most_used_features": list(tree_stats["feature_usage"].keys())[:10],
                "depth_range": f"{tree_stats['depth_stats']['min_depth']}-{tree_stats['depth_stats']['max_depth']}",
                "node_range": f"{tree_stats['node_stats']['min_nodes']}-{tree_stats['node_stats']['max_nodes']}"
            }
            
            logger.info(f"Tree analysis completed: {tree_stats['trees_analyzed']}/{tree_stats['total_trees']} trees analyzed successfully")
            logger.info(f"Average depth: {tree_stats['summary']['avg_depth']}, Average nodes: {tree_stats['summary']['avg_nodes']}")
            
            return tree_stats
            
        except Exception as e:
            logger.error(f"Error extracting tree count and stats: {str(e)}")
            return {"error": str(e)}
    
    def calculate_tree_depths(self) -> Dict[str, Any]:
        """
        Calculate depth for each tree in the Random Forest
        
        This function implements T1.4.2 - Create tree depth calculation for each tree
        
        Returns:
            Dict[str, Any]: Dictionary containing depth information for all trees
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        try:
            logger.info("Calculating tree depths for all trees...")
            
            depth_info = {
                "total_trees": self.model.n_estimators,
                "trees_processed": 0,
                "tree_depths": [],
                "depth_statistics": {
                    "min_depth": float('inf'),
                    "max_depth": 0,
                    "avg_depth": 0,
                    "median_depth": 0,
                    "std_depth": 0
                },
                "depth_distribution": {},
                "depth_categories": {
                    "shallow (1-5)": 0,
                    "medium (6-10)": 0,
                    "deep (11-15)": 0,
                    "very_deep (16+)": 0
                },
                "errors": []
            }
            
            depths = []
            
            # Calculate depth for each tree
            for tree_idx in range(self.model.n_estimators):
                try:
                    tree = self.model.estimators_[tree_idx]
                    tree_structure = tree.tree_
                    tree_depth = tree_structure.max_depth
                    
                    # Store individual tree depth info
                    tree_depth_info = {
                        "tree_id": tree_idx,
                        "depth": tree_depth,
                        "category": self._categorize_depth(tree_depth)
                    }
                    depth_info["tree_depths"].append(tree_depth_info)
                    depths.append(tree_depth)
                    
                    # Update statistics
                    depth_info["depth_statistics"]["min_depth"] = min(depth_info["depth_statistics"]["min_depth"], tree_depth)
                    depth_info["depth_statistics"]["max_depth"] = max(depth_info["depth_statistics"]["max_depth"], tree_depth)
                    
                    # Update distribution
                    depth_key = str(tree_depth)
                    depth_info["depth_distribution"][depth_key] = depth_info["depth_distribution"].get(depth_key, 0) + 1
                    
                    # Update categories
                    category = self._categorize_depth(tree_depth)
                    depth_info["depth_categories"][category] += 1
                    
                    depth_info["trees_processed"] += 1
                    
                except Exception as tree_error:
                    logger.warning(f"Error calculating depth for tree {tree_idx}: {str(tree_error)}")
                    depth_info["errors"].append({
                        "tree_id": tree_idx,
                        "error": str(tree_error)
                    })
            
            # Calculate final statistics
            if depths:
                depth_info["depth_statistics"]["avg_depth"] = round(np.mean(depths), 2)
                depth_info["depth_statistics"]["median_depth"] = round(np.median(depths), 2)
                depth_info["depth_statistics"]["std_depth"] = round(np.std(depths), 2)
            
            # Sort depth distribution by depth value
            depth_info["depth_distribution"] = dict(sorted(
                depth_info["depth_distribution"].items(),
                key=lambda x: int(x[0])
            ))
            
            logger.info(f"Tree depth calculation completed: {depth_info['trees_processed']}/{depth_info['total_trees']} trees processed")
            logger.info(f"Depth range: {depth_info['depth_statistics']['min_depth']}-{depth_info['depth_statistics']['max_depth']}")
            logger.info(f"Average depth: {depth_info['depth_statistics']['avg_depth']}")
            
            return depth_info
            
        except Exception as e:
            logger.error(f"Error calculating tree depths: {str(e)}")
            return {"error": str(e)}
    
    def extract_feature_importance_for_trees(self) -> Dict[str, Any]:
        """
        Extract feature importance for each tree in the Random Forest
        
        This function implements T1.4.3 - Extract feature importance for each tree
        
        Returns:
            Dict[str, Any]: Dictionary containing feature importance information for all trees
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        try:
            logger.info("Extracting feature importance for all trees...")
            
            importance_info = {
                "total_trees": self.model.n_estimators,
                "trees_processed": 0,
                "feature_names": self.feature_names,
                "tree_feature_importances": [],
                "aggregated_importance": {},
                "importance_statistics": {
                    "most_important_features": {},
                    "least_important_features": {},
                    "feature_usage_frequency": {},
                    "average_importance_per_feature": {}
                },
                "ensemble_feature_importance": {},
                "errors": []
            }
            
            # Initialize aggregated importance tracking
            for feature_name in self.feature_names:
                importance_info["aggregated_importance"][feature_name] = []
                importance_info["importance_statistics"]["feature_usage_frequency"][feature_name] = 0
            
            # Extract feature importance for each tree
            for tree_idx in range(self.model.n_estimators):
                try:
                    tree = self.model.estimators_[tree_idx]
                    
                    # Get feature importances for this tree
                    if hasattr(tree, 'feature_importances_'):
                        tree_importances = tree.feature_importances_
                        
                        # Create feature importance mapping for this tree
                        tree_importance_dict = {}
                        for feature_idx, importance in enumerate(tree_importances):
                            if feature_idx < len(self.feature_names):
                                feature_name = self.feature_names[feature_idx]
                                tree_importance_dict[feature_name] = float(importance)
                                
                                # Track aggregated importance
                                importance_info["aggregated_importance"][feature_name].append(importance)
                                
                                # Count feature usage (non-zero importance)
                                if importance > 0:
                                    importance_info["importance_statistics"]["feature_usage_frequency"][feature_name] += 1
                        
                        # Store tree-specific importance info
                        tree_importance_info = {
                            "tree_id": tree_idx,
                            "feature_importances": tree_importance_dict,
                            "top_features": self._get_top_features(tree_importance_dict, 5),
                            "total_importance": sum(tree_importance_dict.values()),
                            "non_zero_features": sum(1 for imp in tree_importance_dict.values() if imp > 0)
                        }
                        importance_info["tree_feature_importances"].append(tree_importance_info)
                        
                    else:
                        logger.warning(f"Tree {tree_idx} does not have feature_importances_ attribute")
                        importance_info["errors"].append({
                            "tree_id": tree_idx,
                            "error": "No feature_importances_ attribute"
                        })
                    
                    importance_info["trees_processed"] += 1
                    
                except Exception as tree_error:
                    logger.warning(f"Error extracting feature importance for tree {tree_idx}: {str(tree_error)}")
                    importance_info["errors"].append({
                        "tree_id": tree_idx,
                        "error": str(tree_error)
                    })
            
            # Calculate aggregated statistics
            for feature_name in self.feature_names:
                importances = importance_info["aggregated_importance"][feature_name]
                if importances:
                    importance_info["importance_statistics"]["average_importance_per_feature"][feature_name] = {
                        "mean": float(np.mean(importances)),
                        "std": float(np.std(importances)),
                        "min": float(np.min(importances)),
                        "max": float(np.max(importances)),
                        "median": float(np.median(importances))
                    }
                else:
                    importance_info["importance_statistics"]["average_importance_per_feature"][feature_name] = {
                        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0
                    }
            
            # Get ensemble feature importance (from the model itself)
            if hasattr(self.model, 'feature_importances_'):
                ensemble_importances = self.model.feature_importances_
                for feature_idx, importance in enumerate(ensemble_importances):
                    if feature_idx < len(self.feature_names):
                        feature_name = self.feature_names[feature_idx]
                        importance_info["ensemble_feature_importance"][feature_name] = float(importance)
            
            # Identify most and least important features
            avg_importances = {
                name: stats["mean"] 
                for name, stats in importance_info["importance_statistics"]["average_importance_per_feature"].items()
            }
            
            # Sort by average importance
            sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
            importance_info["importance_statistics"]["most_important_features"] = dict(sorted_features[:10])
            importance_info["importance_statistics"]["least_important_features"] = dict(sorted_features[-10:])
            
            logger.info(f"Feature importance extraction completed: {importance_info['trees_processed']}/{importance_info['total_trees']} trees processed")
            logger.info(f"Top 3 most important features: {list(importance_info['importance_statistics']['most_important_features'].keys())[:3]}")
            
            return importance_info
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return {"error": str(e)}
    
    def calculate_node_counts_per_tree(self) -> Dict[str, Any]:
        """
        Calculate node counts for each tree in the Random Forest
        
        This function implements T1.4.4 - Calculate node counts per tree
        
        Returns:
            Dict[str, Any]: Dictionary containing node count information for all trees
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        try:
            logger.info("Calculating node counts for all trees...")
            
            node_count_info = {
                "total_trees": self.model.n_estimators,
                "trees_processed": 0,
                "tree_node_counts": [],
                "node_statistics": {
                    "min_nodes": float('inf'),
                    "max_nodes": 0,
                    "avg_nodes": 0,
                    "median_nodes": 0,
                    "std_nodes": 0,
                    "total_nodes_all_trees": 0
                },
                "node_distribution": {},
                "node_categories": {
                    "small (1-25)": 0,
                    "medium (26-75)": 0,
                    "large (76-150)": 0,
                    "very_large (151+)": 0
                },
                "node_type_breakdown": {
                    "total_internal_nodes": 0,
                    "total_leaf_nodes": 0,
                    "avg_internal_nodes_per_tree": 0,
                    "avg_leaf_nodes_per_tree": 0
                },
                "errors": []
            }
            
            node_counts = []
            internal_node_counts = []
            leaf_node_counts = []
            
            # Calculate node counts for each tree
            for tree_idx in range(self.model.n_estimators):
                try:
                    tree = self.model.estimators_[tree_idx]
                    tree_structure = tree.tree_
                    
                    # Get basic node counts
                    total_nodes = tree_structure.node_count
                    
                    # Calculate internal and leaf nodes
                    # Leaf nodes have children_left == -1
                    leaf_nodes = np.sum(tree_structure.children_left == -1)
                    internal_nodes = total_nodes - leaf_nodes
                    
                    # Store individual tree node info
                    tree_node_info = {
                        "tree_id": tree_idx,
                        "total_nodes": total_nodes,
                        "internal_nodes": internal_nodes,
                        "leaf_nodes": leaf_nodes,
                        "category": self._categorize_node_count(total_nodes),
                        "node_density": round(total_nodes / (tree_structure.max_depth + 1), 2) if tree_structure.max_depth > 0 else 0,
                        "leaf_ratio": round(leaf_nodes / total_nodes, 3) if total_nodes > 0 else 0
                    }
                    node_count_info["tree_node_counts"].append(tree_node_info)
                    
                    # Track counts for statistics
                    node_counts.append(total_nodes)
                    internal_node_counts.append(internal_nodes)
                    leaf_node_counts.append(leaf_nodes)
                    
                    # Update statistics
                    node_count_info["node_statistics"]["min_nodes"] = min(node_count_info["node_statistics"]["min_nodes"], total_nodes)
                    node_count_info["node_statistics"]["max_nodes"] = max(node_count_info["node_statistics"]["max_nodes"], total_nodes)
                    node_count_info["node_statistics"]["total_nodes_all_trees"] += total_nodes
                    
                    # Update node type breakdown
                    node_count_info["node_type_breakdown"]["total_internal_nodes"] += internal_nodes
                    node_count_info["node_type_breakdown"]["total_leaf_nodes"] += leaf_nodes
                    
                    # Update distribution (grouped by ranges)
                    node_range = self._get_node_range(total_nodes)
                    node_count_info["node_distribution"][node_range] = node_count_info["node_distribution"].get(node_range, 0) + 1
                    
                    # Update categories
                    category = self._categorize_node_count(total_nodes)
                    node_count_info["node_categories"][category] += 1
                    
                    node_count_info["trees_processed"] += 1
                    
                except Exception as tree_error:
                    logger.warning(f"Error calculating node count for tree {tree_idx}: {str(tree_error)}")
                    node_count_info["errors"].append({
                        "tree_id": tree_idx,
                        "error": str(tree_error)
                    })
            
            # Calculate final statistics
            if node_counts:
                node_count_info["node_statistics"]["avg_nodes"] = round(np.mean(node_counts), 2)
                node_count_info["node_statistics"]["median_nodes"] = round(np.median(node_counts), 2)
                node_count_info["node_statistics"]["std_nodes"] = round(np.std(node_counts), 2)
            
            if internal_node_counts:
                node_count_info["node_type_breakdown"]["avg_internal_nodes_per_tree"] = round(np.mean(internal_node_counts), 2)
            
            if leaf_node_counts:
                node_count_info["node_type_breakdown"]["avg_leaf_nodes_per_tree"] = round(np.mean(leaf_node_counts), 2)
            
            # Sort node distribution by range
            range_order = ["1-10", "11-25", "26-50", "51-100", "101-200", "200+"]
            ordered_distribution = {}
            for range_key in range_order:
                if range_key in node_count_info["node_distribution"]:
                    ordered_distribution[range_key] = node_count_info["node_distribution"][range_key]
            node_count_info["node_distribution"] = ordered_distribution
            
            # Add summary information
            node_count_info["summary"] = {
                "total_trees_analyzed": node_count_info["trees_processed"],
                "analysis_success_rate": round(node_count_info["trees_processed"] / node_count_info["total_trees"] * 100, 2),
                "avg_nodes_per_tree": node_count_info["node_statistics"]["avg_nodes"],
                "total_nodes_in_forest": node_count_info["node_statistics"]["total_nodes_all_trees"],
                "node_range": f"{node_count_info['node_statistics']['min_nodes']}-{node_count_info['node_statistics']['max_nodes']}",
                "most_common_node_range": max(node_count_info["node_distribution"].items(), key=lambda x: x[1])[0] if node_count_info["node_distribution"] else "N/A",
                "internal_to_leaf_ratio": round(
                    node_count_info["node_type_breakdown"]["total_internal_nodes"] / 
                    node_count_info["node_type_breakdown"]["total_leaf_nodes"], 2
                ) if node_count_info["node_type_breakdown"]["total_leaf_nodes"] > 0 else 0
            }
            
            logger.info(f"Node count calculation completed: {node_count_info['trees_processed']}/{node_count_info['total_trees']} trees processed")
            logger.info(f"Average nodes per tree: {node_count_info['node_statistics']['avg_nodes']}")
            logger.info(f"Total nodes in forest: {node_count_info['node_statistics']['total_nodes_all_trees']}")
            
            return node_count_info
            
        except Exception as e:
            logger.error(f"Error calculating node counts: {str(e)}")
            return {"error": str(e)}

    def _get_top_features(self, feature_importance_dict: Dict[str, float], top_n: int = 5) -> Dict[str, float]:
        """
        Helper function to get top N features by importance
        
        Args:
            feature_importance_dict (Dict[str, float]): Feature importance mapping
            top_n (int): Number of top features to return
            
        Returns:
            Dict[str, float]: Top N features with their importance values
        """
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])

    def _categorize_depth(self, depth: int) -> str:
        """
        Helper function to categorize tree depths
        
        Args:
            depth (int): Tree depth
            
        Returns:
            str: Depth category
        """
        if depth <= 5:
            return "shallow (1-5)"
        elif depth <= 10:
            return "medium (6-10)"
        elif depth <= 15:
            return "deep (11-15)"
        else:
            return "very_deep (16+)"

    def _categorize_node_count(self, node_count: int) -> str:
        """
        Helper function to categorize node counts into size categories
        
        Args:
            node_count (int): Number of nodes in a tree
            
        Returns:
            str: Size category for the node count
        """
        if node_count <= 25:
            return "small (1-25)"
        elif node_count <= 75:
            return "medium (26-75)"
        elif node_count <= 150:
            return "large (76-150)"
        else:
            return "very_large (151+)"

    def _get_node_range(self, node_count: int) -> str:
        """
        Helper function to categorize node counts into ranges
        
        Args:
            node_count (int): Number of nodes in a tree
            
        Returns:
            str: Range category for the node count
        """
        if node_count <= 10:
            return "1-10"
        elif node_count <= 25:
            return "11-25"
        elif node_count <= 50:
            return "26-50"
        elif node_count <= 100:
            return "51-100"
        elif node_count <= 200:
            return "101-200"
        else:
            return "200+"

    def create_tree_metadata_json(self) -> Dict[str, Any]:
        """
        Create comprehensive tree metadata JSON structure combining all extracted information
        
        This function implements T1.4.5 - Create tree metadata JSON structure
        
        Returns:
            Dict[str, Any]: Unified tree metadata structure
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        try:
            logger.info("Creating comprehensive tree metadata JSON structure...")
            
            # Extract all metadata components
            tree_stats = self.extract_tree_count_and_stats()
            depth_info = self.calculate_tree_depths()
            importance_info = self.extract_feature_importance_for_trees()
            node_count_info = self.calculate_node_counts_per_tree()
            
            # Create unified metadata structure
            metadata = {
                "model_info": {
                    "model_type": self.model_info.get("model_type", "RandomForestClassifier"),
                    "n_estimators": self.model_info.get("n_estimators", 100),
                    "n_features": len(self.feature_names),
                    "feature_names": self.feature_names,
                    "classes": self.model_info.get("classes", [0, 1]),
                    "random_state": self.model_info.get("random_state"),
                    "max_depth": self.model_info.get("max_depth"),
                    "min_samples_split": self.model_info.get("min_samples_split"),
                    "min_samples_leaf": self.model_info.get("min_samples_leaf"),
                    "bootstrap": self.model_info.get("bootstrap", True)
                },
                "forest_statistics": {
                    "total_trees": tree_stats.get("total_trees", 0),
                    "trees_analyzed": tree_stats.get("trees_analyzed", 0),
                    "analysis_success_rate": tree_stats.get("summary", {}).get("analysis_success_rate", 0),
                    "total_nodes_all_trees": tree_stats.get("node_stats", {}).get("total_nodes", 0),
                    "avg_depth": tree_stats.get("summary", {}).get("avg_depth", 0),
                    "avg_nodes": tree_stats.get("summary", {}).get("avg_nodes", 0),
                    "depth_range": tree_stats.get("summary", {}).get("depth_range", "0-0"),
                    "node_range": tree_stats.get("summary", {}).get("node_range", "0-0")
                },
                "depth_analysis": {
                    "statistics": depth_info.get("depth_statistics", {}),
                    "distribution": depth_info.get("depth_distribution", {}),
                    "categories": depth_info.get("depth_categories", {}),
                    "individual_depths": [
                        {
                            "tree_id": tree["tree_id"],
                            "depth": tree["depth"],
                            "category": tree["category"]
                        }
                        for tree in depth_info.get("tree_depths", [])
                    ]
                },
                "node_analysis": {
                    "statistics": node_count_info.get("node_statistics", {}),
                    "distribution": node_count_info.get("node_distribution", {}),
                    "categories": node_count_info.get("node_categories", {}),
                    "type_breakdown": node_count_info.get("node_type_breakdown", {}),
                    "individual_counts": [
                        {
                            "tree_id": tree["tree_id"],
                            "total_nodes": tree["total_nodes"],
                            "internal_nodes": tree["internal_nodes"],
                            "leaf_nodes": tree["leaf_nodes"],
                            "category": tree["category"],
                            "node_density": tree["node_density"],
                            "leaf_ratio": tree["leaf_ratio"]
                        }
                        for tree in node_count_info.get("tree_node_counts", [])
                    ]
                },
                "feature_importance": {
                    "ensemble_importance": importance_info.get("ensemble_feature_importance", {}),
                    "most_important_features": importance_info.get("importance_statistics", {}).get("most_important_features", {}),
                    "feature_usage_frequency": importance_info.get("importance_statistics", {}).get("feature_usage_frequency", {}),
                    "average_importance_per_feature": importance_info.get("importance_statistics", {}).get("average_importance_per_feature", {}),
                    "individual_tree_importance": [
                        {
                            "tree_id": tree["tree_id"],
                            "top_features": tree["top_features"],
                            "total_importance": tree["total_importance"],
                            "non_zero_features": tree["non_zero_features"]
                        }
                        for tree in importance_info.get("tree_feature_importances", [])
                    ]
                },
                "individual_trees": [],
                "generation_info": {
                    "generated_at": None,  # Will be set when called
                    "processing_time": None,  # Will be calculated
                    "errors": {
                        "depth_errors": len(depth_info.get("errors", [])),
                        "node_count_errors": len(node_count_info.get("errors", [])),
                        "importance_errors": len(importance_info.get("errors", [])),
                        "total_errors": (
                            len(depth_info.get("errors", [])) + 
                            len(node_count_info.get("errors", [])) + 
                            len(importance_info.get("errors", []))
                        )
                    }
                }
            }
            
            # Create individual tree metadata by combining all sources
            logger.info("Combining individual tree metadata...")
            
            # Create lookup dictionaries for efficient access
            depth_lookup = {tree["tree_id"]: tree for tree in depth_info.get("tree_depths", [])}
            node_lookup = {tree["tree_id"]: tree for tree in node_count_info.get("tree_node_counts", [])}
            importance_lookup = {tree["tree_id"]: tree for tree in importance_info.get("tree_feature_importances", [])}
            
            # Combine data for each tree
            for tree_summary in tree_stats.get("tree_summaries", []):
                tree_id = tree_summary["tree_id"]
                
                # Get corresponding data from other analyses
                depth_data = depth_lookup.get(tree_id, {})
                node_data = node_lookup.get(tree_id, {})
                importance_data = importance_lookup.get(tree_id, {})
                
                # Create unified tree metadata
                tree_metadata = {
                    "tree_id": tree_id,
                    "basic_info": {
                        "depth": tree_summary.get("depth", depth_data.get("depth", 0)),
                        "total_nodes": tree_summary.get("nodes", node_data.get("total_nodes", 0)),
                        "internal_nodes": tree_summary.get("internal_nodes", node_data.get("internal_nodes", 0)),
                        "leaf_nodes": tree_summary.get("leaf_nodes", node_data.get("leaf_nodes", 0)),
                        "features_used": tree_summary.get("features_used", 0)
                    },
                    "categorization": {
                        "depth_category": depth_data.get("category", "unknown"),
                        "node_category": node_data.get("category", "unknown")
                    },
                    "metrics": {
                        "node_density": node_data.get("node_density", 0),
                        "leaf_ratio": node_data.get("leaf_ratio", 0),
                        "total_importance": importance_data.get("total_importance", 0),
                        "non_zero_features": importance_data.get("non_zero_features", 0)
                    },
                    "feature_importance": {
                        "top_features": importance_data.get("top_features", {}),
                        "feature_count": importance_data.get("non_zero_features", 0)
                    }
                }
                
                metadata["individual_trees"].append(tree_metadata)
            
            # Sort individual trees by tree_id for consistency
            metadata["individual_trees"].sort(key=lambda x: x["tree_id"])
            
            # Add summary statistics
            metadata["summary"] = {
                "total_trees_with_complete_data": len(metadata["individual_trees"]),
                "data_completeness_rate": len(metadata["individual_trees"]) / metadata["forest_statistics"]["total_trees"] * 100,
                "most_common_depth_category": max(
                    metadata["depth_analysis"]["categories"].items(), 
                    key=lambda x: x[1]
                )[0] if metadata["depth_analysis"]["categories"] else "unknown",
                "most_common_node_category": max(
                    metadata["node_analysis"]["categories"].items(), 
                    key=lambda x: x[1]
                )[0] if metadata["node_analysis"]["categories"] else "unknown",
                "feature_diversity": {
                    "total_unique_features_used": len([f for f, count in tree_stats.get("feature_usage", {}).items() if count > 0]),
                    "most_used_feature": max(tree_stats.get("feature_usage", {}).items(), key=lambda x: x[1])[0] if tree_stats.get("feature_usage") else "unknown",
                    "avg_features_per_tree": sum(tree["basic_info"]["features_used"] for tree in metadata["individual_trees"]) / len(metadata["individual_trees"]) if metadata["individual_trees"] else 0
                }
            }
            
            logger.info(f"Tree metadata JSON structure created successfully with {len(metadata['individual_trees'])} trees")
            logger.info(f"Data completeness rate: {metadata['summary']['data_completeness_rate']:.2f}%")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating tree metadata JSON: {str(e)}")
            return {"error": str(e)}

    def verify_model_structure(self) -> Dict[str, Any]:
        """
        Verify the model structure and tree access
        
        Returns:
            Dict[str, Any]: Dictionary containing verification results
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
            
        try:
            verification = {
                "model_loaded": True,
                "has_estimators": hasattr(self.model, 'estimators_'),
                "n_estimators": self.model.n_estimators,
                "estimators_accessible": False,
                "trees_accessible": False,
                "sample_tree_info": None
            }
            
            # Check if estimators are accessible
            if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
                verification["estimators_accessible"] = True
                
                # Check if individual trees are accessible
                first_tree = self.model.estimators_[0]
                if hasattr(first_tree, 'tree_'):
                    verification["trees_accessible"] = True
                    
                    # Get sample tree information
                    tree_structure = first_tree.tree_
                    verification["sample_tree_info"] = {
                        "node_count": tree_structure.node_count,
                        "max_depth": tree_structure.max_depth,
                        "has_children_left": hasattr(tree_structure, 'children_left'),
                        "has_children_right": hasattr(tree_structure, 'children_right'),
                        "has_feature": hasattr(tree_structure, 'feature'),
                        "has_threshold": hasattr(tree_structure, 'threshold'),
                        "has_value": hasattr(tree_structure, 'value')
                    }
            
            logger.info(f"Model structure verification completed: {verification}")
            return verification
            
        except Exception as e:
            logger.error(f"Error during model structure verification: {str(e)}")
            return {"error": str(e)}

# Convenience functions for easy access
def load_random_forest_model(model_path: str = "data/random_forest_model.pkl") -> RandomForestModelLoader:
    """
    Load the Random Forest model and return a configured loader
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        RandomForestModelLoader: Configured model loader
    """
    loader = RandomForestModelLoader(model_path)
    if loader.load_model():
        loader.extract_model_info()
        return loader
    else:
        raise Exception(f"Failed to load model from {model_path}")

def get_model_summary(model_path: str = "data/random_forest_model.pkl") -> Dict[str, Any]:
    """
    Get a quick summary of the model
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Dict[str, Any]: Model summary
    """
    try:
        loader = load_random_forest_model(model_path)
        summary = {
            "model_info": loader.model_info,
            "verification": loader.verify_model_structure(),
            "tree_statistics": loader.extract_tree_count_and_stats(),
            "sample_prediction": loader.test_prediction()
        }
        return summary
    except Exception as e:
        logger.error(f"Error getting model summary: {str(e)}")
        return {"error": str(e)}

def get_tree_count_and_stats(model_path: str = "data/random_forest_model.pkl") -> Dict[str, Any]:
    """
    Convenience function to get tree count and statistics
    
    This function implements T1.4.1 - Build function to extract tree count and basic stats
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Dict[str, Any]: Tree count and statistics
    """
    try:
        loader = load_random_forest_model(model_path)
        return loader.extract_tree_count_and_stats()
    except Exception as e:
        logger.error(f"Error getting tree count and stats: {str(e)}")
        return {"error": str(e)}

def get_tree_depths(model_path: str = "data/random_forest_model.pkl") -> Dict[str, Any]:
    """
    Convenience function to get tree depth calculations
    
    This function implements T1.4.2 - Create tree depth calculation for each tree
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Dict[str, Any]: Tree depth information
    """
    try:
        loader = load_random_forest_model(model_path)
        return loader.calculate_tree_depths()
    except Exception as e:
        logger.error(f"Error getting tree depths: {str(e)}")
        return {"error": str(e)}

def get_feature_importance_for_trees(model_path: str = "data/random_forest_model.pkl") -> Dict[str, Any]:
    """
    Convenience function to get feature importance for each tree
    
    This function implements T1.4.3 - Extract feature importance for each tree
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Dict[str, Any]: Feature importance information
    """
    try:
        loader = load_random_forest_model(model_path)
        return loader.extract_feature_importance_for_trees()
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return {"error": str(e)}

def get_node_counts_per_tree(model_path: str = "data/random_forest_model.pkl") -> Dict[str, Any]:
    """
    Convenience function to get node counts for each tree
    
    This function implements T1.4.4 - Calculate node counts per tree
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Dict[str, Any]: Node count information for all trees
    """
    try:
        loader = load_random_forest_model(model_path)
        return loader.calculate_node_counts_per_tree()
    except Exception as e:
        logger.error(f"Error getting node counts: {str(e)}")
        return {"error": str(e)}

def get_tree_metadata_json(model_path: str = "data/random_forest_model.pkl") -> Dict[str, Any]:
    """
    Convenience function to get comprehensive tree metadata JSON structure
    
    This function implements T1.4.5 - Create tree metadata JSON structure
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Dict[str, Any]: Unified tree metadata structure
    """
    try:
        loader = load_random_forest_model(model_path)
        return loader.create_tree_metadata_json()
    except Exception as e:
        logger.error(f"Error getting tree metadata JSON: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test the model loader
    try:
        model_path = "data/random_forest_model.pkl"
        print(f"Testing model loader with {model_path}")
        
        summary = get_model_summary(model_path)
        print("Model Summary:")
        print(f"- Model Type: {summary.get('model_info', {}).get('model_type', 'Unknown')}")
        print(f"- Number of Trees: {summary.get('model_info', {}).get('n_estimators', 'Unknown')}")
        print(f"- Number of Features: {len(summary.get('model_info', {}).get('feature_names', []))}")
        print(f"- Trees Accessible: {summary.get('verification', {}).get('trees_accessible', False)}")
        
    except Exception as e:
        print(f"Error: {e}")
