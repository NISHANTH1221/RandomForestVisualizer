"""
Random Forest Visualization Backend API

This FastAPI application provides endpoints for:
- Loading and analyzing Random Forest models
- Extracting tree metadata and structure
- Making predictions with individual tree outputs
- Tracking decision paths through specific trees
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# Import our model modules
from models.model_loader import RandomForestModelLoader
from models.tree_extractor import TreeStructureExtractor
from models.prediction_service import PredictionService
from models.decision_path_tracker import DecisionPathTracker
from models.feature_encoding_service import FeatureEncodingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model loader and extractor instances
model_loader = None
tree_extractor = None

def initialize_model():
    """Initialize the model loader and tree extractor"""
    global model_loader, tree_extractor
    
    try:
        model_path = Path(__file__).parent / "data" / "random_forest_model.pkl"
        logger.info(f"Loading model from {model_path}")
        
        model_loader = RandomForestModelLoader(str(model_path))
        if model_loader.load_model():
            model_loader.extract_model_info()
            tree_extractor = TreeStructureExtractor(model_loader.model)
            logger.info("Model and tree extractor initialized successfully")
            return True
        else:
            logger.error("Failed to load model")
            return False
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False

# Initialize FastAPI app
app = FastAPI(
    title="Random Forest Visualization API",
    description="Backend API for Random Forest model visualization and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to initialize model
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting up Random Forest Visualization API...")
    if initialize_model():
        logger.info("Model initialized successfully on startup")
    else:
        logger.warning("Failed to initialize model on startup - will retry on first request")

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    error_message: str
    billing_state: str
    card_funding: str
    card_network: str
    card_issuer: str

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    ensemble_prediction: float
    individual_predictions: List[Dict[str, Any]]
    confidence_interval: Optional[Dict[str, float]] = None

class DecisionPathRequest(BaseModel):
    """Request model for decision path tracking"""
    tree_id: int
    error_message: str
    billing_state: str
    card_funding: str
    card_network: str
    card_issuer: str

class TreeMetadata(BaseModel):
    """Model for tree metadata"""
    tree_id: int
    depth: int
    node_count: int
    feature_importance: Dict[str, float]

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Random Forest Visualization API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "trees": "/api/trees",
            "predict": "/api/predict",
            "decision_path": "/api/decision-path",
            "feature_options": "/api/feature-options"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

# API endpoints
@app.get("/api/trees")
async def get_trees():
    """
    Get metadata for all trees in the Random Forest
    
    This endpoint implements T2.5.1 - Create /api/trees endpoint (get all tree metadata)
    
    Returns:
        Dict[str, Any]: Metadata for all trees in the forest
    """
    global model_loader, tree_extractor
    
    # Initialize model if not already done
    if model_loader is None or tree_extractor is None:
        if not initialize_model():
            raise HTTPException(status_code=500, detail="Failed to initialize model")
    
    try:
        logger.info("Extracting metadata for all trees")
        
        # Get basic model information
        model_info = model_loader.model_info
        
        # Extract metadata for all trees
        all_trees_metadata = []
        
        for tree_id in range(model_loader.model.n_estimators):
            try:
                # Get tree structure for metadata calculation
                tree_estimator = model_loader.model.estimators_[tree_id]
                tree_structure = tree_estimator.tree_
                
                # Calculate tree metadata
                tree_metadata = {
                    "tree_id": tree_id,
                    "depth": int(tree_structure.max_depth),
                    "node_count": int(tree_structure.node_count),
                    "leaf_count": int(np.sum(tree_structure.children_left == -1)),
                    "internal_node_count": int(tree_structure.node_count - np.sum(tree_structure.children_left == -1)),
                    "feature_importance": {},  # Simplified for now
                    "tree_statistics": {
                        "avg_samples_per_leaf": float(np.mean([tree_structure.n_node_samples[i] for i in range(tree_structure.node_count) if tree_structure.children_left[i] == -1])),
                        "max_samples_in_node": int(np.max(tree_structure.n_node_samples)),
                        "min_samples_in_node": int(np.min(tree_structure.n_node_samples)),
                        "avg_impurity": float(np.mean(tree_structure.impurity)),
                        "tree_balance_score": 0.5  # Simplified for now
                    }
                }
                
                all_trees_metadata.append(tree_metadata)
                
            except Exception as e:
                logger.warning(f"Failed to extract metadata for tree {tree_id}: {str(e)}")
                # Add minimal metadata for failed trees
                all_trees_metadata.append({
                    "tree_id": tree_id,
                    "error": str(e),
                    "depth": 0,
                    "node_count": 0,
                    "leaf_count": 0,
                    "internal_node_count": 0,
                    "feature_importance": {},
                    "tree_statistics": {}
                })
        
        # Calculate forest-level statistics
        successful_trees = [tree for tree in all_trees_metadata if "error" not in tree]
        
        forest_statistics = {
            "total_trees": len(all_trees_metadata),
            "successful_extractions": len(successful_trees),
            "failed_extractions": len(all_trees_metadata) - len(successful_trees),
            "average_depth": float(np.mean([tree["depth"] for tree in successful_trees])) if successful_trees else 0,
            "average_node_count": float(np.mean([tree["node_count"] for tree in successful_trees])) if successful_trees else 0,
            "depth_range": {
                "min": int(np.min([tree["depth"] for tree in successful_trees])) if successful_trees else 0,
                "max": int(np.max([tree["depth"] for tree in successful_trees])) if successful_trees else 0
            },
            "node_count_range": {
                "min": int(np.min([tree["node_count"] for tree in successful_trees])) if successful_trees else 0,
                "max": int(np.max([tree["node_count"] for tree in successful_trees])) if successful_trees else 0
            }
        }
        
        # Aggregate feature importance across all trees
        aggregated_feature_importance = {}
        for tree in successful_trees:
            for feature, importance in tree["feature_importance"].items():
                if feature not in aggregated_feature_importance:
                    aggregated_feature_importance[feature] = []
                aggregated_feature_importance[feature].append(importance)
        
        # Calculate average feature importance
        avg_feature_importance = {}
        for feature, importances in aggregated_feature_importance.items():
            avg_feature_importance[feature] = float(np.mean(importances))
        
        # Sort by importance
        sorted_feature_importance = dict(sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        response = {
            "success": True,
            "model_info": {
                "n_estimators": model_info["n_estimators"],
                "n_features": model_info.get("n_features_in", len(model_info.get("feature_names", []))),
                "feature_names": model_info.get("feature_names", []),
                "model_type": model_info.get("model_type", "RandomForestClassifier")
            },
            "forest_statistics": forest_statistics,
            "aggregated_feature_importance": sorted_feature_importance,
            "trees": all_trees_metadata,
            "extraction_info": {
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0",
                "total_trees_processed": len(all_trees_metadata)
            }
        }
        
        logger.info(f"Successfully extracted metadata for {len(successful_trees)}/{len(all_trees_metadata)} trees")
        return response
        
    except Exception as e:
        logger.error(f"Error extracting trees metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract trees metadata: {str(e)}")

@app.get("/api/trees/{tree_id}")
async def get_tree_structure(tree_id: int, depth: Optional[int] = None):
    """
    Get detailed structure for a specific tree with optional depth limitation
    
    This endpoint implements T2.1.5 - Create API endpoint /api/trees/{id}
    Enhanced with depth parameter support
    
    Args:
        tree_id (int): ID of the tree to extract (0 to n_estimators-1)
        depth (Optional[int]): Maximum depth to fetch (None for full tree, must be >= 1)
        
    Returns:
        Dict[str, Any]: Tree structure with nodes and metadata, limited by depth if specified
    """
    global tree_extractor
    
    # Initialize model if not already done
    if tree_extractor is None:
        if not initialize_model():
            raise HTTPException(status_code=500, detail="Failed to initialize model")
    
    try:
        # Validate tree ID
        if not tree_extractor.validate_tree_id(tree_id):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid tree ID: {tree_id}. Must be between 0 and {tree_extractor.model.n_estimators-1}"
            )
        
        # Validate depth parameter
        if depth is not None:
            if depth < 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid depth parameter: {depth}. Must be >= 1 or None for full tree"
                )
            if depth > 20:  # Reasonable upper limit to prevent performance issues
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid depth parameter: {depth}. Must be <= 20 for performance reasons"
                )
        
        # Extract tree structure with depth limitation
        logger.info(f"Extracting structure for tree {tree_id} with depth limit: {depth}")
        
        if depth is not None:
            tree_structure = tree_extractor.convert_tree_to_json_with_depth_limit(tree_id, depth)
        else:
            tree_structure = tree_extractor.convert_tree_to_json(tree_id)
        
        # Add extraction timestamp
        from datetime import datetime
        tree_structure["extraction_timestamp"] = datetime.now().isoformat()
        
        # Add API metadata
        api_response = {
            "success": True,
            "tree_id": tree_id,
            "depth_limit": depth,
            "extraction_info": {
                "timestamp": tree_structure["extraction_timestamp"],
                "format_version": tree_structure["format_version"],
                "api_version": "1.1.0",  # Updated version to indicate depth support
                "depth_limited": tree_structure["metadata"].get("is_depth_limited", False),
                "effective_max_depth": tree_structure["metadata"].get("effective_max_depth", tree_structure["metadata"]["max_depth"]),
                "truncated_nodes": tree_structure["statistics"].get("truncated_nodes", 0)
            },
            "tree_data": tree_structure
        }
        
        total_nodes = tree_structure["statistics"].get("filtered_total_nodes", tree_structure["metadata"]["total_nodes"])
        original_nodes = tree_structure["statistics"].get("original_total_nodes", tree_structure["metadata"]["total_nodes"])
        
        if depth is not None:
            logger.info(f"Successfully extracted tree {tree_id} with depth limit {depth}: {total_nodes}/{original_nodes} nodes")
        else:
            logger.info(f"Successfully extracted tree {tree_id}: {total_nodes} nodes")
        
        return api_response
        
    except ValueError as e:
        logger.error(f"Validation error for tree {tree_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error extracting tree {tree_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract tree structure: {str(e)}")

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """
    Make predictions using all trees in the Random Forest
    
    This endpoint implements T2.5.2 - Create /api/predict endpoint (POST with parameters)
    
    Args:
        request (PredictionRequest): Prediction request with input parameters
        
    Returns:
        Dict[str, Any]: Predictions from all trees with ensemble result
    """
    global model_loader
    
    # Initialize model if not already done
    if model_loader is None:
        if not initialize_model():
            raise HTTPException(status_code=500, detail="Failed to initialize model")
    
    try:
        logger.info(f"Making predictions for input: {request.dict()}")
        
        # Create prediction service
        prediction_service = PredictionService(model_loader.model)
        
        # Convert request to input dictionary
        input_data = {
            "error_message": request.error_message,
            "billing_state": request.billing_state,
            "card_funding": request.card_funding,
            "card_network": request.card_network,
            "card_issuer": request.card_issuer
        }
        
        # Validate input using feature encoding service
        encoding_service = FeatureEncodingService()
        validation_result = encoding_service.validate_input_parameters(input_data)
        
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "Invalid input parameters",
                    "errors": validation_result["errors"],
                    "warnings": validation_result["warnings"],
                    "suggestions": validation_result["suggestions"]
                }
            )
        
        # Use validated data for prediction
        validated_input = validation_result["validated_data"]
        
        # Make predictions with all trees
        batch_result = prediction_service.predict_all_trees(validated_input)
        
        # Format response
        response = {
            "success": True,
            "input_data": {
                "original": input_data,
                "validated": validated_input,
                "validation_warnings": validation_result["warnings"]
            },
            "ensemble_prediction": batch_result["ensemble_prediction"],
            "individual_predictions": batch_result["individual_predictions"],
            "statistics": batch_result["statistics"],
            "performance": batch_result["performance"],
            "prediction_info": {
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0",
                "model_type": batch_result["metadata"]["model_type"],
                "total_trees_used": batch_result["statistics"]["successful_predictions"]
            }
        }
        
        logger.info(f"Prediction completed successfully. Ensemble result: {batch_result['ensemble_prediction']['value']:.4f}")
        logger.info(f"Individual predictions: {batch_result['statistics']['successful_predictions']}/{batch_result['statistics']['total_trees']} trees")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to make predictions: {str(e)}")

@app.post("/api/decision-path")
async def get_decision_path(request: DecisionPathRequest):
    """
    Get decision path for a specific tree and input
    
    This endpoint implements T2.5.3 - Create /api/decision-path endpoint (POST tree_id + parameters)
    
    Args:
        request (DecisionPathRequest): Decision path request with tree ID and input parameters
        
    Returns:
        Dict[str, Any]: Complete decision path through the specified tree
    """
    global model_loader
    
    # Initialize model if not already done
    if model_loader is None:
        if not initialize_model():
            raise HTTPException(status_code=500, detail="Failed to initialize model")
    
    try:
        logger.info(f"Tracking decision path for tree {request.tree_id} with input: {request.dict()}")
        
        # Validate tree ID
        if request.tree_id < 0 or request.tree_id >= model_loader.model.n_estimators:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tree ID: {request.tree_id}. Must be between 0 and {model_loader.model.n_estimators-1}"
            )
        
        # Create decision path tracker
        path_tracker = DecisionPathTracker(model_loader.model)
        
        # Convert request to input dictionary
        input_data = {
            "error_message": request.error_message,
            "billing_state": request.billing_state,
            "card_funding": request.card_funding,
            "card_network": request.card_network,
            "card_issuer": request.card_issuer
        }
        
        # Validate input using feature encoding service
        encoding_service = FeatureEncodingService()
        validation_result = encoding_service.validate_input_parameters(input_data)
        
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid input parameters",
                    "errors": validation_result["errors"],
                    "warnings": validation_result["warnings"],
                    "suggestions": validation_result["suggestions"]
                }
            )
        
        # Use validated data for path tracking
        validated_input = validation_result["validated_data"]
        
        # Track decision path
        path_result = path_tracker.track_decision_path(request.tree_id, validated_input)
        
        # Format response
        response = {
            "success": True,
            "tree_id": request.tree_id,
            "input_data": {
                "original": input_data,
                "validated": validated_input,
                "validation_warnings": validation_result["warnings"]
            },
            "decision_path": path_result["decision_path"],
            "path_metadata": path_result["path_metadata"],
            "path_summary": path_result["path_summary"],
            "validation": path_result["validation"],
            "api_info": {
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0",
                "path_extraction_time_ms": path_result["path_metadata"]["path_statistics"]["traversal_time_ms"],
                "nodes_visited": len(path_result["decision_path"])
            }
        }
        
        final_prediction = path_result["path_metadata"]["prediction_result"]["final_prediction"]
        path_valid = path_result["validation"]["path_valid"]
        
        logger.info(f"Decision path tracking completed for tree {request.tree_id}")
        logger.info(f"Path: {len(path_result['decision_path'])} nodes, Final prediction: {final_prediction}, Valid: {path_valid}")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error tracking decision path for tree {request.tree_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track decision path: {str(e)}")

@app.post("/api/tree-visualization")
async def get_tree_visualization(request: DecisionPathRequest, max_depth: int = 4):
    """
    Get tree visualization with limited depth and highlighted decision path
    
    Args:
        request (DecisionPathRequest): Decision path request with tree ID and input parameters
        max_depth (int): Maximum depth to show in visualization (default: 4)
        
    Returns:
        Dict[str, Any]: Tree structure with limited depth and highlighted path
    """
    global model_loader
    
    # Initialize model if not already done
    if model_loader is None:
        if not initialize_model():
            raise HTTPException(status_code=500, detail="Failed to initialize model")
    
    try:
        logger.info(f"Getting tree visualization for tree {request.tree_id} with max depth {max_depth}")
        
        # Validate tree ID
        if request.tree_id < 0 or request.tree_id >= model_loader.model.n_estimators:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tree ID: {request.tree_id}. Must be between 0 and {model_loader.model.n_estimators-1}"
            )
        
        # Validate max_depth
        if max_depth < 1 or max_depth > 10:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid max_depth: {max_depth}. Must be between 1 and 10"
            )
        
        # Create decision path tracker
        path_tracker = DecisionPathTracker(model_loader.model)
        
        # Convert request to input dictionary
        input_data = {
            "error_message": request.error_message,
            "billing_state": request.billing_state,
            "card_funding": request.card_funding,
            "card_network": request.card_network,
            "card_issuer": request.card_issuer
        }
        
        # Validate input using feature encoding service
        encoding_service = FeatureEncodingService()
        validation_result = encoding_service.validate_input_parameters(input_data)
        
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid input parameters",
                    "errors": validation_result["errors"],
                    "warnings": validation_result["warnings"],
                    "suggestions": validation_result["suggestions"]
                }
            )
        
        # Use validated data for tree extraction
        validated_input = validation_result["validated_data"]
        
        # Extract tree with limited depth and highlighted path
        tree_result = path_tracker.extract_tree_with_limited_depth_and_path(
            request.tree_id, validated_input, max_depth
        )
        
        # Format response
        response = {
            "success": True,
            "tree_id": request.tree_id,
            "max_depth_limit": max_depth,
            "input_data": {
                "original": input_data,
                "validated": validated_input,
                "validation_warnings": validation_result["warnings"]
            },
            "tree_structure": tree_result,
            "api_info": {
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0",
                "extraction_time_ms": tree_result["extraction_info"]["extraction_time_ms"],
                "nodes_in_view": tree_result["tree_metadata"]["total_nodes_in_limited_view"],
                "nodes_on_path": tree_result["tree_metadata"]["nodes_on_path"]
            }
        }
        
        logger.info(f"Tree visualization completed for tree {request.tree_id}")
        logger.info(f"Nodes in view: {tree_result['tree_metadata']['total_nodes_in_limited_view']}, "
                   f"Nodes on path: {tree_result['tree_metadata']['nodes_on_path']}")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting tree visualization for tree {request.tree_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tree visualization: {str(e)}")

@app.get("/api/feature-options")
async def get_feature_options():
    """
    Get available options for dropdown fields
    
    This endpoint implements T2.5.4 - Create /api/feature-options endpoint (get dropdown options)
    
    Returns:
        Dict[str, Any]: Available options for all dropdown fields
    """
    try:
        logger.info("Retrieving feature options for dropdown fields")
        
        # Create feature encoding service
        encoding_service = FeatureEncodingService()
        
        # Get feature options
        feature_options = encoding_service.get_feature_options()
        
        # Add additional metadata for each field
        enhanced_options = {}
        
        for field_name, options in feature_options.items():
            enhanced_options[field_name] = {
                "options": options,
                "total_count": len(options),
                "field_type": "categorical",
                "required": True,
                "description": _get_field_description(field_name),
                "validation_rules": {
                    "allow_empty": False,
                    "allow_unknown": True,
                    "default_value": options[0] if options else None
                }
            }
        
        # Add time-based feature options
        enhanced_options["time_features"] = {
            "weekday": {
                "options": list(range(1, 8)),  # 1-7 (Monday to Sunday)
                "labels": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                "total_count": 7,
                "field_type": "numeric",
                "required": False,
                "description": "Day of the week (1=Monday, 7=Sunday)",
                "validation_rules": {
                    "min_value": 1,
                    "max_value": 7,
                    "default_value": 1
                }
            },
            "day": {
                "options": list(range(1, 32)),  # 1-31
                "total_count": 31,
                "field_type": "numeric", 
                "required": False,
                "description": "Day of the month (1-31)",
                "validation_rules": {
                    "min_value": 1,
                    "max_value": 31,
                    "default_value": 15
                }
            },
            "hour": {
                "options": list(range(0, 24)),  # 0-23
                "total_count": 24,
                "field_type": "numeric",
                "required": False,
                "description": "Hour of the day (0-23, 24-hour format)",
                "validation_rules": {
                    "min_value": 0,
                    "max_value": 23,
                    "default_value": 10
                }
            }
        }
        
        # Calculate total combinations
        total_combinations = 1
        for field_name, field_info in enhanced_options.items():
            if field_name != "time_features":
                total_combinations *= field_info["total_count"]
        
        # Add business logic suggestions
        business_suggestions = {
            "common_combinations": [
                {
                    "name": "High Risk Transaction",
                    "description": "Typical high-risk transaction pattern",
                    "values": {
                        "error_message": "insufficient_funds",
                        "billing_state": "CA",
                        "card_funding": "credit",
                        "card_network": "visa",
                        "card_issuer": "chase"
                    }
                },
                {
                    "name": "Low Risk Transaction", 
                    "description": "Typical low-risk transaction pattern",
                    "values": {
                        "error_message": "approved",
                        "billing_state": "NY",
                        "card_funding": "debit",
                        "card_network": "mastercard",
                        "card_issuer": "bank_of_america"
                    }
                },
                {
                    "name": "Fraud Suspected",
                    "description": "Transaction flagged for potential fraud",
                    "values": {
                        "error_message": "fraud_suspected",
                        "billing_state": "FL",
                        "card_funding": "credit",
                        "card_network": "amex",
                        "card_issuer": "american_express"
                    }
                }
            ],
            "field_relationships": {
                "card_network_issuer": {
                    "description": "Common card network and issuer combinations",
                    "combinations": {
                        "visa": ["chase", "bank_of_america", "wells_fargo", "citibank"],
                        "mastercard": ["chase", "bank_of_america", "wells_fargo", "citibank"],
                        "amex": ["american_express"],
                        "discover": ["discover"]
                    }
                },
                "error_risk_levels": {
                    "description": "Error messages grouped by risk level",
                    "groups": {
                        "high_risk": ["insufficient_funds", "fraud_suspected", "limit_exceeded", "velocity_exceeded"],
                        "medium_risk": ["card_declined", "expired_card", "invalid_card", "authentication_failed"],
                        "low_risk": ["approved", "processing_error", "network_error"]
                    }
                }
            }
        }
        
        response = {
            "success": True,
            "feature_options": enhanced_options,
            "metadata": {
                "total_fields": len([k for k in enhanced_options.keys() if k != "time_features"]),
                "total_categorical_options": sum(field_info["total_count"] for field_name, field_info in enhanced_options.items() if field_name != "time_features"),
                "total_possible_combinations": total_combinations,
                "encoding_type": "one_hot",
                "feature_vector_length": sum(field_info["total_count"] for field_name, field_info in enhanced_options.items() if field_name != "time_features") + 3  # +3 for time features
            },
            "business_suggestions": business_suggestions,
            "api_info": {
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0",
                "encoding_service_version": "1.0.0"
            }
        }
        
        logger.info(f"Feature options retrieved successfully: {len(enhanced_options)} fields")
        logger.info(f"Total categorical options: {response['metadata']['total_categorical_options']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving feature options: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feature options: {str(e)}")

def _get_field_description(field_name: str) -> str:
    """
    Get human-readable description for a field
    
    Args:
        field_name (str): Name of the field
        
    Returns:
        str: Human-readable description
    """
    descriptions = {
        "error_message": "The error message or status returned by the payment processor",
        "billing_state": "The billing state/province associated with the payment method",
        "card_funding": "The funding type of the payment card (credit, debit, prepaid, etc.)",
        "card_network": "The payment network that processes the card (Visa, Mastercard, etc.)",
        "card_issuer": "The financial institution that issued the payment card"
    }
    
    return descriptions.get(field_name, f"Options for {field_name.replace('_', ' ').title()}")

# Global exception handlers for standardized error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with standardized format"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return {
        "success": False,
        "error": {
            "type": "http_error",
            "status_code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    logger.warning(f"404 Not Found: {request.url.path}")
    return {
        "success": False,
        "error": {
            "type": "not_found",
            "status_code": 404,
            "message": "Endpoint not found",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "available_endpoints": [
                "/docs", "/health", "/api/trees", "/api/predict", 
                "/api/decision-path", "/api/feature-options"
            ]
        }
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors"""
    error_id = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.error(f"Internal server error [{error_id}]: {exc}")
    return {
        "success": False,
        "error": {
            "type": "internal_error",
            "status_code": 500,
            "message": "Internal server error occurred",
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    }

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return {
        "success": False,
        "error": {
            "type": "validation_error",
            "status_code": 400,
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
