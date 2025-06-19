'use client';

import * as React from 'react';
import { getTreeStructure } from '@/services/api';

interface TreeNode {
  node_id: number;
  feature_name?: string;
  threshold?: number;
  is_leaf: boolean;
  prediction?: number;
  samples: number;
  gini_impurity: number;
  depth: number;
  children?: TreeNode[];
  is_truncated?: boolean;
  truncated_children_count?: number;
}

interface SimpleTreeViewerProps {
  treeId: number;
  treeData?: TreeNode;
  onBack?: () => void;
}

interface ApiTreeNode {
  node_id: number;
  feature_name?: string;
  threshold?: number;
  is_leaf: boolean;
  prediction?: number;
  samples: number;
  gini_impurity: number;
  depth: number;
  is_truncated?: boolean;
  truncated_children_count?: number;
  children?: {
    left?: ApiTreeNode;
    right?: ApiTreeNode;
  };
}

interface ApiTreeResponse {
  success: boolean;
  tree_id: number;
  depth_limit?: number;
  extraction_info: {
    timestamp: string;
    format_version: string;
    api_version: string;
    depth_limited?: boolean;
    effective_max_depth?: number;
    truncated_nodes?: number;
  };
  tree_data: {
    tree_id: number;
    root_node: ApiTreeNode;
    metadata: {
      total_nodes: number;
      max_depth: number;
      effective_max_depth?: number;
      is_depth_limited?: boolean;
      depth_limit?: number;
    };
    statistics: {
      filtered_total_nodes?: number;
      original_total_nodes?: number;
      truncated_nodes?: number;
    };
    format_version: string;
    extraction_timestamp: string;
  };
}

// Convert API tree structure to our TreeNode format
function convertApiTreeToTreeNode(apiNode: ApiTreeNode): TreeNode {
  // Try different possible field names for gini impurity
  const giniValue = apiNode.gini_impurity || 
                   (apiNode as any).impurity || 
                   (apiNode as any).gini || 
                   (apiNode as any).weighted_n_node_samples || 
                   0;

  const node: TreeNode = {
    node_id: apiNode.node_id,
    feature_name: apiNode.feature_name,
    threshold: apiNode.threshold,
    is_leaf: apiNode.is_leaf,
    prediction: apiNode.prediction,
    samples: apiNode.samples,
    gini_impurity: giniValue,
    depth: apiNode.depth,
    is_truncated: apiNode.is_truncated || false,
    truncated_children_count: apiNode.truncated_children_count || 0,
    children: []
  };

  // Debug log to see what fields are actually available
  if (typeof window !== 'undefined') {
    console.log('API Node fields:', Object.keys(apiNode));
    console.log('Gini value found:', giniValue);
    console.log('Full API node:', apiNode);
  }

  // Add children if node has them
  if (apiNode.children) {
    const children: TreeNode[] = [];
    
    if (apiNode.children.left) {
      children.push(convertApiTreeToTreeNode(apiNode.children.left));
    }
    
    if (apiNode.children.right) {
      children.push(convertApiTreeToTreeNode(apiNode.children.right));
    }
    
    if (children.length > 0) {
      node.children = children;
    }
  }

  return node;
}

function SimpleTreeNode({ node }: { node: TreeNode }) {
  const isLeaf = node.is_leaf;
  
  const getNodeColor = () => {
    if (isLeaf) {
      const prediction = node.prediction || 0;
      if (prediction >= 0.7) return 'bg-green-50 border-green-300 text-green-800 shadow-green-100';
      if (prediction >= 0.5) return 'bg-yellow-50 border-yellow-300 text-yellow-800 shadow-yellow-100';
      return 'bg-red-50 border-red-300 text-red-800 shadow-red-100';
    }
    
    return 'bg-blue-50 border-blue-300 text-blue-800 shadow-blue-100';
  };

  const getNodeIcon = () => {
    if (isLeaf) {
      return (
        <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
          <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        </div>
      );
    }
    
    return (
      <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center">
        <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
    );
  };

  return (
    <div className="flex flex-col items-center relative">
      {/* Node */}
      <div className={`relative border-2 rounded-xl p-4 m-2 min-w-[160px] max-w-[200px] text-center shadow-lg hover:shadow-xl transition-all duration-300 ${getNodeColor()}`}>
        {/* Node Header */}
        <div className="flex items-center justify-center gap-2 mb-3">
          {getNodeIcon()}
          <div className="font-bold text-sm">
            {isLeaf ? 'Leaf Node' : `Node ${node.node_id}`}
          </div>
        </div>

        {/* Node Content */}
        {!isLeaf ? (
          <div className="space-y-2 text-xs">
            <div className="bg-white/50 rounded-lg p-2">
              <div className="font-semibold text-gray-700 mb-1">Feature:</div>
              <div className="font-mono text-xs break-words">{node.feature_name}</div>
            </div>
            <div className="bg-white/50 rounded-lg p-2">
              <div className="font-semibold text-gray-700 mb-1">Threshold:</div>
              <div className="font-mono">≤ {node.threshold?.toFixed(3)}</div>
            </div>
          </div>
        ) : (
          <div className="text-center">
            <div className="bg-white/50 rounded-lg p-2">
              <div className="font-semibold text-gray-700 mb-1">Prediction</div>
              <div className={`text-lg font-bold ${
                (node.prediction || 0) >= 0.7 ? 'text-green-600' :
                (node.prediction || 0) >= 0.5 ? 'text-yellow-600' :
                'text-red-600'
              }`}>
                {((node.prediction || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        {/* Node Statistics */}
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-center">
              <div className="font-bold">{node.samples}</div>
              <div className="text-gray-600">Samples</div>
            </div>
            <div className="text-center">
              <div className="font-bold">{(node.gini_impurity || 0).toFixed(3)}</div>
              <div className="text-gray-600">Gini</div>
            </div>
          </div>
        </div>
      </div>

      {/* Children */}
      {node.children && node.children.length > 0 && (
        <div className="relative mt-4">
          {/* Vertical line from parent */}
          <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-0.5 h-6 bg-gray-400"></div>
          
          {/* Horizontal line connecting children */}
          {node.children.length === 2 && (
            <div className="absolute top-6 left-0 right-0 h-0.5 bg-gray-400"></div>
          )}
          
          {/* Children container with proper centering */}
          <div className="flex justify-center items-start pt-6" style={{ minWidth: '600px' }}>
            {node.children.map((child, index) => (
              <div key={child.node_id} className="flex flex-col items-center relative" style={{ 
                flex: '1 1 0%',
                minWidth: '250px'
              }}>
                {/* Vertical line to child */}
                <div className="w-0.5 h-6 bg-gray-400 mb-2 mx-auto"></div>
                
                {/* Branch label */}
                <div className={`mb-2 px-3 py-1 rounded-full text-xs font-bold mx-auto ${
                  index === 0 
                    ? 'bg-green-100 text-green-700 border border-green-300' 
                    : 'bg-red-100 text-red-700 border border-red-300'
                }`}>
                  {index === 0 ? 'YES (≤)' : 'NO (>)'}
                </div>
                
                {/* Child node */}
                <div className="flex justify-center w-full">
                  <SimpleTreeNode node={child} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Helper functions
function getMaxDepth(node: TreeNode): number {
  if (!node.children || node.children.length === 0) {
    return node.depth;
  }
  return Math.max(...node.children.map(child => getMaxDepth(child)));
}

function countNodes(node: TreeNode): number {
  if (!node.children || node.children.length === 0) {
    return 1;
  }
  return 1 + node.children.reduce((sum, child) => sum + countNodes(child), 0);
}

export default function SimpleTreeViewer({ treeId, treeData, onBack }: SimpleTreeViewerProps) {
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [apiTreeData, setApiTreeData] = React.useState<TreeNode | null>(null);
  const [treeMetadata, setTreeMetadata] = React.useState<any>(null);
  const [maxDepth, setMaxDepth] = React.useState(3); // Default depth, adjustable up to 5

  // Fetch tree data from API with depth parameter
  const fetchTreeData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response: ApiTreeResponse = await getTreeStructure(treeId, maxDepth);
      
      if (response.success && response.tree_data?.root_node) {
        const convertedTree = convertApiTreeToTreeNode(response.tree_data.root_node);
        setApiTreeData(convertedTree);
        setTreeMetadata(response.tree_data);
      } else {
        setError('Failed to load tree data from API');
      }
    } catch (err) {
      console.error('Error fetching tree data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch tree data');
    } finally {
      setLoading(false);
    }
  };

  // Fetch tree data when component mounts
  React.useEffect(() => {
    if (!treeData && treeId !== undefined) {
      fetchTreeData();
    }
  }, [treeId, treeData]);

  // Use provided treeData or fetched apiTreeData
  const displayData = treeData || apiTreeData;

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-gray-900 dark:via-slate-900 dark:to-indigo-950 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20">
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  Loading Tree Structure
                </h3>
                <p className="text-gray-500 dark:text-gray-400">
                  Fetching tree {treeId} data with max depth {maxDepth}...
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-gray-900 dark:via-slate-900 dark:to-indigo-950 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20">
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Error Loading Tree
              </h3>
              <p className="text-gray-500 dark:text-gray-400 mb-4">
                {error}
              </p>
              <div className="flex gap-4 justify-center">
                <button
                  onClick={fetchTreeData}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-xl transition-colors"
                >
                  Try Again
                </button>
                {onBack && (
                  <button
                    onClick={onBack}
                    className="bg-gray-500 hover:bg-gray-600 text-white font-medium py-2 px-4 rounded-xl transition-colors"
                  >
                    Back to Grid
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!displayData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-gray-900 dark:via-slate-900 dark:to-indigo-950 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20">
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-1.447-.894L15 4m0 13V4m-6 3l6-3" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                No Tree Data Available
              </h3>
              <p className="text-gray-500 dark:text-gray-400 mb-4">
                Unable to load tree structure for Tree {treeId}
              </p>
              {onBack && (
                <button
                  onClick={onBack}
                  className="bg-gray-500 hover:bg-gray-600 text-white font-medium py-2 px-4 rounded-xl transition-colors"
                >
                  Back to Grid
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Get metadata for display
  const isDepthLimited = treeMetadata?.metadata?.is_depth_limited || false;
  const effectiveMaxDepth = treeMetadata?.metadata?.effective_max_depth || getMaxDepth(displayData);
  const originalMaxDepth = treeMetadata?.metadata?.max_depth || getMaxDepth(displayData);
  const truncatedNodes = treeMetadata?.statistics?.truncated_nodes || 0;
  const totalNodes = treeMetadata?.statistics?.filtered_total_nodes || countNodes(displayData);
  const originalNodes = treeMetadata?.statistics?.original_total_nodes || totalNodes;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-gray-900 dark:via-slate-900 dark:to-indigo-950 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20 mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Tree {treeId} Structure
              </h2>
              <p className="text-gray-600 dark:text-gray-300 mt-2">
                Interactive decision tree visualization showing the complete structure
              </p>
            </div>
            
            {onBack && (
              <button
                onClick={onBack}
                className="bg-gradient-to-r from-gray-500 to-gray-600 hover:from-gray-600 hover:to-gray-700 text-white font-medium py-3 px-6 rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                ← Back to Grid
              </button>
            )}
          </div>

          {/* Tree Info */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-xl p-4 border border-blue-200/50">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {treeId}
              </div>
              <div className="text-gray-600 dark:text-gray-400 font-medium">Tree ID</div>
            </div>
            <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-xl p-4 border border-green-200/50">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {displayData.samples}
              </div>
              <div className="text-gray-600 dark:text-gray-400 font-medium">Total Samples</div>
            </div>
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-xl p-4 border border-purple-200/50">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {getMaxDepth(displayData)}
              </div>
              <div className="text-gray-600 dark:text-gray-400 font-medium">Max Depth</div>
            </div>
            <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-xl p-4 border border-orange-200/50">
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {countNodes(displayData)}
              </div>
              <div className="text-gray-600 dark:text-gray-400 font-medium">Total Nodes</div>
            </div>
          </div>
        </div>

        {/* Depth Control */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Tree Depth Control
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Adjust the maximum depth to explore different levels of the tree
              </p>
            </div>
            <button
              onClick={fetchTreeData}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-2 px-4 rounded-lg transition-colors"
            >
              {loading ? 'Loading...' : 'Refresh Tree'}
            </button>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Max Depth: {maxDepth}
                </label>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  Range: 1-5
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                value={maxDepth}
                onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 slider"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((maxDepth - 1) / 4) * 100}%, #e5e7eb ${((maxDepth - 1) / 4) * 100}%, #e5e7eb 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                <span>1</span>
                <span>2</span>
                <span>3</span>
                <span>4</span>
                <span>5</span>
              </div>
            </div>
          </div>
        </div>

        {/* Tree Visualization */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20">
          <div className="mb-6">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              Decision Tree Structure (Depth: {maxDepth})
            </h3>
            <p className="text-gray-600 dark:text-gray-300 text-sm">
              Follow the path from root to leaf nodes. Green indicates high success rate, yellow medium, and red low success rate.
            </p>
          </div>
          
          <div className="overflow-x-auto overflow-y-auto max-h-[80vh] bg-gradient-to-br from-gray-50 to-white rounded-xl p-6 border border-gray-200">
            <div className="min-w-max flex justify-center">
              <SimpleTreeNode node={displayData} />
            </div>
          </div>

          {/* Legend */}
          <div className="mt-6 p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl border border-gray-200">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Legend</h4>
            <div className="flex flex-wrap gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-100 border border-green-300 rounded"></div>
                <span>High Success (≥70%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-yellow-100 border border-yellow-300 rounded"></div>
                <span>Medium Success (50-69%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-100 border border-red-300 rounded"></div>
                <span>Low Success (&lt;50%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-blue-100 border border-blue-300 rounded"></div>
                <span>Decision Node</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
