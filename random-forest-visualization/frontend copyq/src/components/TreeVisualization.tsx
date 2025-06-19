'use client';

import * as React from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { getTreeVisualization, TreeVisualizationResponse, TreeVisualizationNode, PredictionRequest } from '@/services/api';

interface TreeVisualizationProps {
  treeId?: number;
  formData?: PredictionRequest;
  maxDepth?: number;
}

interface TreeNodeComponentProps {
  node: TreeVisualizationNode;
  visualizationHints: any;
  onNodeClick?: (node: TreeVisualizationNode) => void;
}

function TreeNodeComponent({ node, visualizationHints, onNodeClick }: TreeNodeComponentProps) {
  const isLeaf = node.is_leaf;
  const isOnPath = node.is_on_path;
  const isBeyondMaxDepth = node.is_beyond_max_depth;
  
  const getNodeColor = () => {
    if (isOnPath) {
      if (isBeyondMaxDepth) {
        return `border-3 border-amber-500 bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/30 dark:to-amber-800/30 shadow-xl ring-2 ring-amber-300/50`;
      }
      return `border-3 border-blue-500 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 shadow-xl ring-2 ring-blue-300/50`;
    }
    
    if (isLeaf) {
      const prediction = node.prediction || 0;
      if (prediction >= 0.7) return 'border-2 border-green-500 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 shadow-lg';
      if (prediction >= 0.5) return 'border-2 border-yellow-500 bg-gradient-to-br from-yellow-50 to-yellow-100 dark:from-yellow-900/20 dark:to-yellow-800/20 shadow-lg';
      return 'border-2 border-red-500 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 shadow-lg';
    }
    
    return 'border-2 border-gray-300 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 hover:from-gray-100 hover:to-gray-200 dark:hover:from-gray-600 dark:hover:to-gray-700 shadow-md hover:shadow-lg';
  };

  const getNodeIcon = () => {
    if (isLeaf) {
      return (
        <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
          <svg className="w-5 h-5 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        </div>
      );
    }
    
    return (
      <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
        <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
    );
  };

  const handleClick = () => {
    if (onNodeClick) {
      onNodeClick(node);
    }
  };

  return (
    <div className="flex flex-col items-center relative">
      {/* Node */}
      <div 
        className={`relative rounded-xl p-4 min-w-[180px] max-w-[220px] cursor-pointer transition-all duration-300 hover:scale-105 transform ${getNodeColor()}`}
        onClick={handleClick}
      >
        {/* Path indicator */}
        {isOnPath && (
          <div className="absolute -top-2 -right-2 w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg animate-pulse"></div>
        )}
        
        {/* Depth indicator */}
        {isBeyondMaxDepth && (
          <div className="absolute -top-2 -left-2 w-4 h-4 bg-amber-500 rounded-full border-2 border-white shadow-lg">
            <div className="absolute inset-0 rounded-full bg-amber-500 animate-ping opacity-75"></div>
          </div>
        )}

        {/* Node Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            {getNodeIcon()}
            <div>
              <div className="font-bold text-sm text-gray-900 dark:text-white">
                {isLeaf ? 'Leaf Node' : `Node ${node.node_id}`}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Depth {node.depth}
              </div>
            </div>
          </div>
          
          {/* Decision indicator for path nodes */}
          {isOnPath && node.decision_taken && (
            <div className={`px-2 py-1 rounded-full text-xs font-bold ${
              node.decision_taken === 'left' 
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' 
                : 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300'
            }`}>
              {node.decision_taken === 'left' ? '← LEFT' : 'RIGHT →'}
            </div>
          )}
        </div>

        {/* Node Content */}
        {!isLeaf ? (
          <div className="space-y-2 text-sm">
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-2">
              <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">Split Condition:</div>
              <div className="font-mono text-xs text-gray-900 dark:text-white break-all">
                {node.feature_name} ≤ {node.threshold?.toFixed(3)}
              </div>
            </div>
            
            {isOnPath && node.feature_value !== undefined && (
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-2">
                <div className="font-medium text-blue-700 dark:text-blue-300 mb-1">Input Value:</div>
                <div className="font-mono text-xs text-blue-900 dark:text-blue-100">
                  {node.feature_value.toFixed(3)}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center">
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Final Prediction</div>
              <div className={`text-2xl font-bold ${
                (node.prediction || 0) >= 0.7 ? 'text-green-600 dark:text-green-400' :
                (node.prediction || 0) >= 0.5 ? 'text-yellow-600 dark:text-yellow-400' :
                'text-red-600 dark:text-red-400'
              }`}>
                {((node.prediction || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        {/* Node Statistics */}
        <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-center">
              <div className="font-bold text-gray-900 dark:text-white">{node.samples}</div>
              <div className="text-gray-500 dark:text-gray-400">Samples</div>
            </div>
            <div className="text-center">
              <div className="font-bold text-gray-900 dark:text-white">{node.gini_impurity.toFixed(3)}</div>
              <div className="text-gray-500 dark:text-gray-400">Gini</div>
            </div>
          </div>
        </div>
      </div>

      {/* Children with pyramid-style tree structure */}
      {node.children && node.children.length > 0 && (
        <div className="relative mt-12">
          {/* Main vertical line from parent */}
          <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-1 h-16 bg-gradient-to-b from-gray-400 to-gray-600 dark:from-gray-500 dark:to-gray-300 rounded-full shadow-sm"></div>
          
          {/* T-junction for branching */}
          <div className="absolute top-16 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-gray-600 dark:bg-gray-300 rounded-full border-2 border-white shadow-lg"></div>
          
          {/* Diagonal connection lines to children */}
          {node.children.length === 2 && (
            <>
              {/* Left diagonal line */}
              <div 
                className="absolute top-20 left-1/2 w-1 bg-gradient-to-br from-blue-400 to-blue-600 dark:from-blue-500 dark:to-blue-300 rounded-full shadow-sm"
                style={{
                  height: '80px',
                  transformOrigin: 'top center',
                  transform: 'translateX(-50%) rotate(-35deg) translateX(-120px)'
                }}
              ></div>
              
              {/* Right diagonal line */}
              <div 
                className="absolute top-20 left-1/2 w-1 bg-gradient-to-bl from-purple-400 to-purple-600 dark:from-purple-500 dark:to-purple-300 rounded-full shadow-sm"
                style={{
                  height: '80px',
                  transformOrigin: 'top center',
                  transform: 'translateX(-50%) rotate(35deg) translateX(120px)'
                }}
              ></div>
            </>
          )}
          
          {/* Children container with pyramid spacing */}
          <div className="flex justify-center items-start pt-24" style={{ gap: `${Math.max(240, node.depth * 60)}px` }}>
            {node.children.map((child, index) => (
              <div key={child.node_id} className="flex flex-col items-center relative">
                {/* Connection point at child */}
                <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-gray-600 dark:bg-gray-300 rounded-full border-2 border-white shadow-md"></div>
                
                {/* Branch label with enhanced styling and positioning */}
                <div className={`absolute -top-16 left-1/2 transform -translate-x-1/2 px-4 py-2 rounded-full text-sm font-bold shadow-lg border-2 border-white ${
                  index === 0 
                    ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white' 
                    : 'bg-gradient-to-r from-purple-500 to-purple-600 text-white'
                }`}>
                  {index === 0 ? '≤ TRUE' : '> FALSE'}
                </div>
                
                {/* Child node */}
                <TreeNodeComponent 
                  node={child} 
                  visualizationHints={visualizationHints}
                  onNodeClick={onNodeClick}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function TreeVisualization({ treeId: propTreeId, formData: propFormData, maxDepth = 4 }: TreeVisualizationProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [treeVisualization, setTreeVisualization] = React.useState<TreeVisualizationResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [selectedNode, setSelectedNode] = React.useState<TreeVisualizationNode | null>(null);
  const [zoomLevel, setZoomLevel] = React.useState(1);

  // Get tree ID and form data from props, URL params, or session storage
  const treeId = propTreeId || parseInt(searchParams?.get('tree') || '0');
  const [formData, setFormData] = React.useState<PredictionRequest | null>(propFormData || null);

  React.useEffect(() => {
    // Try to get form data from session storage if not provided
    if (!formData) {
      try {
        const storedData = sessionStorage.getItem('currentPrediction');
        if (storedData) {
          const parsed = JSON.parse(storedData);
          setFormData(parsed.formData);
        }
      } catch (error) {
        console.error('Failed to parse stored prediction data:', error);
      }
    }
  }, [formData]);

  React.useEffect(() => {
    if (treeId !== undefined && formData) {
      fetchTreeVisualization();
    }
  }, [treeId, formData, maxDepth]);

  const fetchTreeVisualization = async () => {
    if (!formData || treeId === undefined) return;

    setLoading(true);
    setError(null);
    
    try {
      const response = await getTreeVisualization({
        ...formData,
        tree_id: treeId
      }, maxDepth);
      
      setTreeVisualization(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch tree visualization');
    } finally {
      setLoading(false);
    }
  };

  const handleNodeClick = (node: TreeVisualizationNode) => {
    setSelectedNode(node);
  };

  const closeNodeDetails = () => {
    setSelectedNode(null);
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Loading Tree Visualization
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            Extracting tree structure with max depth {maxDepth}...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Error Loading Tree Visualization
          </h3>
          <p className="text-gray-500 dark:text-gray-400 mb-4">
            {error}
          </p>
          <button
            onClick={fetchTreeVisualization}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!treeVisualization || !formData) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-1.447-.894L15 4m0 13V4m-6 3l6-3" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No Tree Visualization Data
          </h3>
          <p className="text-gray-500 dark:text-gray-400 mb-4">
            Please make a prediction first, then click on a tree to view its visualization.
          </p>
          <button
            onClick={() => router.push('/predict')}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
          >
            Go to Prediction
          </button>
        </div>
      </div>
    );
  }

  const { tree_structure } = treeVisualization;

  return (
    <div className="space-y-6">
      {/* Header with Tree Info */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Tree Visualization - Tree {tree_structure.tree_id}
            </h2>
            <p className="text-gray-600 dark:text-gray-300">
              Interactive tree view with decision path highlighted (Max depth: {tree_structure.max_depth_limit})
            </p>
          </div>
          
          <div className="text-right">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {tree_structure.path_highlighting?.final_prediction?.final_prediction 
                ? (tree_structure.path_highlighting.final_prediction.final_prediction * 100).toFixed(1) + '%'
                : 'N/A'}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Final Prediction
            </div>
          </div>
        </div>

        {/* Tree Metadata */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="font-medium text-gray-900 dark:text-white">
              {tree_structure.tree_metadata.total_nodes_in_limited_view}
            </div>
            <div className="text-gray-500 dark:text-gray-400">Nodes in View</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="font-medium text-gray-900 dark:text-white">
              {tree_structure.tree_metadata.nodes_on_path}
            </div>
            <div className="text-gray-500 dark:text-gray-400">Nodes on Path</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="font-medium text-gray-900 dark:text-white">
              {tree_structure.tree_metadata.max_depth_in_view}
            </div>
            <div className="text-gray-500 dark:text-gray-400">Max Depth Shown</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="font-medium text-gray-900 dark:text-white">
              {tree_structure.tree_metadata.path_coverage.toFixed(1)}%
            </div>
            <div className="text-gray-500 dark:text-gray-400">Path Coverage</div>
          </div>
        </div>

        {/* Legend */}
        <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <h4 className="font-medium text-gray-900 dark:text-white mb-2">Legend:</h4>
          <div className="flex flex-wrap gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-500 bg-blue-50 rounded"></div>
              <span className="text-gray-700 dark:text-gray-300">Decision Path</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border border-green-500 bg-green-50 rounded"></div>
              <span className="text-gray-700 dark:text-gray-300">High Prediction (≥70%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border border-yellow-500 bg-yellow-50 rounded"></div>
              <span className="text-gray-700 dark:text-gray-300">Medium Prediction (50-70%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border border-red-500 bg-red-50 rounded"></div>
              <span className="text-gray-700 dark:text-gray-300">Low Prediction (&lt;50%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-amber-500 bg-amber-50 rounded"></div>
              <span className="text-gray-700 dark:text-gray-300">Beyond Max Depth</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.back()}
              className="bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
            >
              Back to Results
            </button>
            <button
              onClick={() => router.push(`/decision-path?tree=${tree_structure.tree_id}`)}
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
            >
              View Decision Path
            </button>
          </div>
          
          {/* Zoom Controls */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Zoom:</span>
            <button
              onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.1))}
              disabled={zoomLevel <= 0.5}
              className="bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed text-gray-700 font-medium py-1 px-3 rounded-md transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
              </svg>
            </button>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300 min-w-[60px] text-center">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={() => setZoomLevel(Math.min(2.0, zoomLevel + 0.1))}
              disabled={zoomLevel >= 2.0}
              className="bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed text-gray-700 font-medium py-1 px-3 rounded-md transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
            <button
              onClick={() => setZoomLevel(1)}
              className="bg-blue-100 hover:bg-blue-200 text-blue-700 font-medium py-1 px-3 rounded-md transition-colors text-sm"
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Tree Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
          Tree Structure
        </h3>
        
        <div className="overflow-x-auto overflow-y-auto max-h-[80vh]">
          <div 
            className="min-w-max p-4 transition-transform duration-300 ease-in-out origin-top-left"
            style={{ 
              transform: `scale(${zoomLevel})`,
              transformOrigin: 'top left'
            }}
          >
            {tree_structure.root && (
              <TreeNodeComponent 
                node={tree_structure.root} 
                visualizationHints={tree_structure.visualization_hints}
                onNodeClick={handleNodeClick}
              />
            )}
          </div>
        </div>
      </div>

      {/* Node Details Modal */}
      {selectedNode && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Node {selectedNode.node_id} Details
              </h3>
              <button
                onClick={closeNodeDetails}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Type: </span>
                  <span className="text-gray-900 dark:text-white">
                    {selectedNode.is_leaf ? 'Leaf Node' : 'Decision Node'}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Depth: </span>
                  <span className="text-gray-900 dark:text-white">{selectedNode.depth}</span>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Samples: </span>
                  <span className="text-gray-900 dark:text-white">{selectedNode.samples}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Gini: </span>
                  <span className="text-gray-900 dark:text-white">{selectedNode.gini_impurity.toFixed(4)}</span>
                </div>
              </div>
              
              {!selectedNode.is_leaf && (
                <>
                  <div>
                    <span className="font-medium text-gray-700 dark:text-gray-300">Feature: </span>
                    <span className="text-gray-900 dark:text-white">{selectedNode.feature_name}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700 dark:text-gray-300">Threshold: </span>
                    <span className="text-gray-900 dark:text-white font-mono">
                      {selectedNode.threshold?.toFixed(6)}
                    </span>
                  </div>
                </>
              )}
              
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Prediction: </span>
                <span className={`font-bold ${
                  (selectedNode.prediction || 0) >= 0.7 ? 'text-green-600 dark:text-green-400' :
                  (selectedNode.prediction || 0) >= 0.5 ? 'text-yellow-600 dark:text-yellow-400' :
                  'text-red-600 dark:text-red-400'
                }`}>
                  {((selectedNode.prediction || 0) * 100).toFixed(2)}%
                </span>
              </div>
              
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">On Decision Path: </span>
                <span className={`font-medium ${
                  selectedNode.is_on_path ? 'text-blue-600 dark:text-blue-400' : 'text-gray-600 dark:text-gray-400'
                }`}>
                  {selectedNode.is_on_path ? 'Yes' : 'No'}
                </span>
              </div>
              
              {selectedNode.is_beyond_max_depth && (
                <div className="p-2 bg-amber-50 dark:bg-amber-900/20 rounded border border-amber-200 dark:border-amber-800">
                  <span className="text-amber-800 dark:text-amber-200 text-xs">
                    This node is beyond the maximum depth limit but shown because it's on the decision path.
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
