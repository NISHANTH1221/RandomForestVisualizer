'use client';

import * as React from 'react';
import { getTreeVisualization, TreeVisualizationResponse, TreeVisualizationNode, PredictionRequest } from '@/services/api';
import EnhancedTreeViewer from './EnhancedTreeViewer';

interface DecisionPathProps {
  treeId?: number;
  formData?: PredictionRequest;
}

// Convert TreeVisualizationNode to our TreeNode interface
function convertTreeNode(node: TreeVisualizationNode): any {
  return {
    node_id: node.node_id,
    feature_name: node.feature_name,
    threshold: node.threshold,
    is_leaf: node.is_leaf,
    prediction: node.prediction,
    success_probability: node.success_probability,
    samples: node.samples,
    gini_impurity: node.gini_impurity,
    impurity: node.impurity,
    depth: node.depth,
    is_on_path: node.is_on_path,
    decision_taken: node.decision_taken,
    feature_value: node.feature_value,
    confidence: node.confidence,
    is_beyond_max_depth: node.is_beyond_max_depth,
    is_left_child: node.is_left_child,
    children: node.children ? node.children.map(convertTreeNode) : undefined
  };
}

export default function DecisionPath({ treeId: propTreeId, formData: propFormData }: DecisionPathProps) {
  const [treeVisualization, setTreeVisualization] = React.useState<TreeVisualizationResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [maxDepth, setMaxDepth] = React.useState(4);
  const [treeId, setTreeId] = React.useState(propTreeId || 1);
  const [formData, setFormData] = React.useState<PredictionRequest | null>(propFormData || null);

  React.useEffect(() => {
    // Get tree ID from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const treeParam = urlParams.get('tree');
    if (treeParam) {
      setTreeId(parseInt(treeParam));
    }

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
    if (treeId && formData) {
      fetchTreeVisualization();
    }
  }, [treeId, formData, maxDepth]);

  const fetchTreeVisualization = async () => {
    if (!formData || !treeId) return;

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

  const handleBack = () => {
    window.history.back();
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Loading Decision Path
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            Extracting tree structure with decision path for Tree {treeId}...
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
            Error Loading Decision Path
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
            No Decision Path Data
          </h3>
          <p className="text-gray-500 dark:text-gray-400 mb-4">
            Please make a prediction first, then click on a tree to view its decision path.
          </p>
          <button
            onClick={() => window.location.href = '/predict'}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
          >
            Go to Prediction
          </button>
        </div>
      </div>
    );
  }

  // Create a set of node IDs that are on the decision path for easy lookup
  const pathNodeIds = new Set(treeVisualization.tree_structure.path_highlighting.highlighted_node_ids);

  // Convert the tree structure to our format
  const convertedTreeData = treeVisualization.tree_structure.root ? convertTreeNode(treeVisualization.tree_structure.root) : undefined;

  const finalPrediction = treeVisualization.tree_structure.path_highlighting?.final_prediction?.final_prediction;

  return (
    <div className="space-y-6">
      {/* Enhanced Tree Viewer with Path Highlighting */}
      <EnhancedTreeViewer
        treeId={treeVisualization.tree_id}
        treeData={convertedTreeData}
        pathNodeIds={pathNodeIds}
        showPathHighlighting={true}
        onBack={handleBack}
        title={`Decision Path - Tree ${treeVisualization.tree_id}`}
        subtitle={`Interactive tree structure showing decision path with ${finalPrediction ? (finalPrediction * 100).toFixed(1) + '% prediction' : 'highlighted path'} (up to ${maxDepth} levels)`}
      />

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label htmlFor="maxDepth" className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Max Depth:
            </label>
            <select
              id="maxDepth"
              value={maxDepth}
              onChange={(e) => setMaxDepth(parseInt(e.target.value))}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              {[2, 3, 4, 5, 6, 7, 8].map(depth => (
                <option key={depth} value={depth}>{depth}</option>
              ))}
            </select>
          </div>
          
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Showing {treeVisualization.tree_structure.tree_metadata.nodes_on_path} nodes on decision path
          </div>
        </div>
      </div>

      {/* Path Summary */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Decision Path Summary
        </h3>
        
        <div className="space-y-4">
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Decisions Made:</h4>
            <div className="space-y-2">
              {treeVisualization.tree_structure.decision_path.filter(node => !node.is_leaf).map((node, index) => (
                <div key={node.node_id} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-sm">
                  <span className="font-medium">{node.feature_name}</span>
                  <span className="text-gray-600 dark:text-gray-300 mx-2">≤ {node.threshold?.toFixed(3)}</span>
                  <span className={`font-medium ${
                    node.decision_taken === 'left' ? 'text-blue-600 dark:text-blue-400' : 'text-purple-600 dark:text-purple-400'
                  }`}>
                    → Go {node.decision_taken} (value: {node.feature_value?.toFixed(3)})
                  </span>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Final Result:</h4>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-sm">
              {treeVisualization.tree_structure.decision_path && treeVisualization.tree_structure.decision_path.length > 0 ? (
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <span className="text-gray-600 dark:text-gray-300">Samples: </span>
                    <span className="font-medium">
                      {treeVisualization.tree_structure.decision_path[treeVisualization.tree_structure.decision_path.length - 1]?.samples || 'N/A'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-300">Gini: </span>
                    <span className="font-medium">
                      {treeVisualization.tree_structure.decision_path[treeVisualization.tree_structure.decision_path.length - 1]?.gini_impurity?.toFixed(3) || 'N/A'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-300">Prediction: </span>
                    <span className={`font-bold text-lg ${
                      (finalPrediction || 0) >= 0.7 ? 'text-green-600 dark:text-green-400' :
                      (finalPrediction || 0) >= 0.5 ? 'text-yellow-600 dark:text-yellow-400' :
                      'text-red-600 dark:text-red-400'
                    }`}>
                      {finalPrediction ? (finalPrediction * 100).toFixed(1) + '%' : 'N/A'}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-gray-500 dark:text-gray-400">
                  No decision path data available
                </div>
              )}
            </div>
          </div>

          {/* Tree Metadata */}
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Tree Statistics:</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <div className="font-medium text-gray-900 dark:text-white">
                  {treeVisualization.tree_structure.tree_metadata.total_nodes_in_limited_view}
                </div>
                <div className="text-gray-500 dark:text-gray-400">Nodes in View</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <div className="font-medium text-gray-900 dark:text-white">
                  {treeVisualization.tree_structure.tree_metadata.nodes_on_path}
                </div>
                <div className="text-gray-500 dark:text-gray-400">Nodes on Path</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <div className="font-medium text-gray-900 dark:text-white">
                  {treeVisualization.tree_structure.tree_metadata.path_coverage.toFixed(1)}%
                </div>
                <div className="text-gray-500 dark:text-gray-400">Path Coverage</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <div className="font-medium text-gray-900 dark:text-white">
                  {treeVisualization.tree_structure.extraction_info.extraction_time_ms.toFixed(2)}ms
                </div>
                <div className="text-gray-500 dark:text-gray-400">Extraction Time</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
