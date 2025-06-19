/**
 * API Service Layer for Random Forest Visualization
 * 
 * This module provides functions to interact with the backend API endpoints.
 * It handles all HTTP requests and response formatting for the frontend.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types for API responses
export interface TreeMetadata {
  tree_id: number;
  depth: number;
  node_count: number;
  leaf_count: number;
  internal_node_count: number;
  feature_importance: Record<string, number>;
  tree_statistics: {
    avg_samples_per_leaf: number;
    max_samples_in_node: number;
    min_samples_in_node: number;
    avg_impurity: number;
    tree_balance_score: number;
  };
  error?: string;
}

export interface ForestStatistics {
  total_trees: number;
  successful_extractions: number;
  failed_extractions: number;
  average_depth: number;
  average_node_count: number;
  depth_range: {
    min: number;
    max: number;
  };
  node_count_range: {
    min: number;
    max: number;
  };
}

export interface TreesResponse {
  success: boolean;
  model_info: {
    n_estimators: number;
    n_features: number;
    feature_names: string[];
    model_type: string;
  };
  forest_statistics: ForestStatistics;
  aggregated_feature_importance: Record<string, number>;
  trees: TreeMetadata[];
  extraction_info: {
    timestamp: string;
    api_version: string;
    total_trees_processed: number;
  };
}

export interface PredictionRequest {
  error_message: string;
  billing_state: string;
  card_funding: string;
  card_network: string;
  card_issuer: string;
  time_features?: [number, number, number]; // [weekday, day, hour]
}

export interface IndividualPrediction {
  tree_id: number;
  prediction: {
    value: number;
    confidence: number;
    success_probability: number;
    prediction_class: number;
    probability?: number[];
  };
  performance: {
    prediction_time_ms: number;
  };
  classification: {
    risk_level: string;
    confidence_level: string;
    prediction_strength: string;
  };
}

export interface PredictionResponse {
  success: boolean;
  input_data: {
    original: PredictionRequest;
    validated: Record<string, any>;
    validation_warnings: string[];
  };
  ensemble_prediction: {
    value: number;
    confidence: number;
    prediction_class: string;
  };
  individual_predictions: IndividualPrediction[];
  statistics: {
    total_trees: number;
    successful_predictions: number;
    failed_predictions: number;
    average_prediction: number;
    prediction_std: number;
    confidence_interval: {
      lower: number;
      upper: number;
      confidence_level: number;
    };
  };
  performance: {
    total_time_ms: number;
    avg_time_per_tree_ms: number;
  };
  prediction_info: {
    timestamp: string;
    api_version: string;
    model_type: string;
    total_trees_used: number;
  };
}

export interface DecisionPathRequest extends PredictionRequest {
  tree_id: number;
}

export interface DecisionNode {
  node_id: number;
  is_leaf: boolean;
  feature_name?: string;
  feature_index?: number;
  threshold?: number;
  decision_condition?: string;
  feature_value?: number;
  decision_taken?: 'left' | 'right';
  samples: number;
  gini_impurity: number;
  impurity?: number;
  class_distribution?: Record<string, number>;
  prediction?: number;
  confidence?: number;
}

export interface DecisionPathResponse {
  success: boolean;
  tree_id: number;
  input_data: {
    original: PredictionRequest;
    validated: Record<string, any>;
    validation_warnings: string[];
  };
  decision_path: DecisionNode[];
  path_metadata: {
    total_nodes_visited: number;
    tree_depth: number;
    path_length: number;
    prediction_result: {
      final_prediction: number;
      confidence: number;
      prediction_class: string;
    };
    path_statistics: {
      traversal_time_ms: number;
      avg_samples_per_node: number;
      avg_gini_per_node: number;
    };
  };
  path_summary: {
    decisions_made: Array<{
      node_id: number;
      feature_name: string;
      condition: string;
      decision: 'left' | 'right';
    }>;
    final_leaf_info: {
      node_id: number;
      samples: number;
      gini: number;
      prediction: number;
    };
  };
  validation: {
    path_valid: boolean;
    consistency_checks: Record<string, boolean>;
  };
  api_info: {
    timestamp: string;
    api_version: string;
    path_extraction_time_ms: number;
    nodes_visited: number;
  };
}

export interface TreeVisualizationNode {
  node_id: number;
  depth: number;
  samples: number;
  impurity: number;
  gini_impurity: number;
  value: number[][];
  is_leaf: boolean;
  is_on_path: boolean;
  is_beyond_max_depth?: boolean;
  children: TreeVisualizationNode[];
  feature_index?: number;
  feature_name?: string;
  threshold?: number;
  left_child_id?: number;
  right_child_id?: number;
  feature_value_from_input?: any;
  decision_condition?: string;
  decision_taken?: 'left' | 'right';
  next_node_on_path?: number;
  feature_value?: number;
  success_probability: number;
  prediction: number;
  confidence: number;
  prediction_probability?: number[];
  prediction_type?: string;
  parent_id?: number;
  is_left_child?: boolean;
}

export interface TreeVisualizationResponse {
  success: boolean;
  tree_id: number;
  max_depth_limit: number;
  input_data: {
    original: PredictionRequest;
    validated: Record<string, any>;
    validation_warnings: string[];
  };
  tree_structure: {
    tree_id: number;
    max_depth_limit: number;
    root: TreeVisualizationNode;
    all_nodes: Record<number, TreeVisualizationNode>;
    decision_path: DecisionNode[];
    tree_metadata: {
      total_nodes_in_limited_view: number;
      leaf_nodes: number;
      internal_nodes: number;
      max_depth_in_view: number;
      nodes_on_path: number;
      nodes_beyond_max_depth: number;
      path_coverage: number;
      original_tree_depth: number;
      original_tree_nodes: number;
    };
    path_highlighting: {
      highlighted_node_ids: number[];
      path_depth: number;
      path_nodes: number[];
      final_prediction: any;
    };
    visualization_hints: {
      layout_type: string;
      highlight_color: string;
      default_color: string;
      leaf_color: string;
      beyond_depth_color: string;
      show_feature_names: boolean;
      show_thresholds: boolean;
      show_sample_counts: boolean;
      show_gini_values: boolean;
      show_depth_indicators: boolean;
      max_depth_limit: number;
    };
    input_data: {
      original_features: Record<string, any>;
      feature_dict: Record<string, any>;
      feature_array_shape: number[];
    };
    extraction_info: {
      extraction_time_ms: number;
      timestamp: number;
      extraction_type: string;
    };
  };
  api_info: {
    timestamp: string;
    api_version: string;
    extraction_time_ms: number;
    nodes_in_view: number;
    nodes_on_path: number;
  };
}

export interface FeatureOption {
  options: string[] | number[];
  total_count: number;
  field_type: 'categorical' | 'numeric';
  required: boolean;
  description: string;
  validation_rules: {
    allow_empty?: boolean;
    allow_unknown?: boolean;
    default_value?: string | number;
    min_value?: number;
    max_value?: number;
  };
  labels?: string[];
}

export interface FeatureOptionsResponse {
  success: boolean;
  feature_options: Record<string, FeatureOption>;
  metadata: {
    total_fields: number;
    total_categorical_options: number;
    total_possible_combinations: number;
    encoding_type: string;
    feature_vector_length: number;
  };
  business_suggestions: {
    common_combinations: Array<{
      name: string;
      description: string;
      values: PredictionRequest;
    }>;
    field_relationships: Record<string, any>;
  };
  api_info: {
    timestamp: string;
    api_version: string;
    encoding_service_version: string;
  };
}

// API Error type
export interface ApiError {
  success: false;
  error: {
    type: string;
    status_code: number;
    message: string;
    timestamp: string;
    path: string;
    error_id?: string;
    available_endpoints?: string[];
  };
}

/**
 * Generic API request function with error handling
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const defaultOptions: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, defaultOptions);
    
    if (!response.ok) {
      // Try to parse error response
      try {
        const errorData = await response.json();
        throw new Error(errorData.error?.message || `HTTP ${response.status}: ${response.statusText}`);
      } catch {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    }

    const data = await response.json();
    return data as T;
  } catch (error) {
    console.error(`API request failed for ${endpoint}:`, error);
    throw error;
  }
}

/**
 * Get metadata for all trees in the Random Forest
 */
export async function getAllTrees(): Promise<TreesResponse> {
  return apiRequest<TreesResponse>('/api/trees');
}

/**
 * Get detailed structure for a specific tree with optional depth limitation
 */
export async function getTreeStructure(treeId: number, depth?: number): Promise<any> {
  const url = depth !== undefined ? `/api/trees/${treeId}?depth=${depth}` : `/api/trees/${treeId}`;
  return apiRequest(url);
}

/**
 * Make predictions using all trees in the Random Forest
 */
export async function makePrediction(request: PredictionRequest): Promise<PredictionResponse> {
  return apiRequest<PredictionResponse>('/api/predict', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Get decision path for a specific tree and input
 */
export async function getDecisionPath(request: DecisionPathRequest): Promise<DecisionPathResponse> {
  return apiRequest<DecisionPathResponse>('/api/decision-path', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Get tree visualization with limited depth and highlighted decision path
 */
export async function getTreeVisualization(request: DecisionPathRequest, maxDepth: number = 4): Promise<TreeVisualizationResponse> {
  return apiRequest<TreeVisualizationResponse>(`/api/tree-visualization?max_depth=${maxDepth}`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Get available options for dropdown fields
 */
export async function getFeatureOptions(): Promise<FeatureOptionsResponse> {
  return apiRequest<FeatureOptionsResponse>('/api/feature-options');
}

/**
 * Health check endpoint
 */
export async function healthCheck(): Promise<{ status: string; message: string }> {
  return apiRequest('/health');
}

/**
 * Get API root information
 */
export async function getApiInfo(): Promise<any> {
  return apiRequest('/');
}
