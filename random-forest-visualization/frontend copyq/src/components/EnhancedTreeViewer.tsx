'use client';

import * as React from 'react';

interface TreeNode {
  node_id: number;
  feature_name?: string;
  threshold?: number;
  is_leaf: boolean;
  prediction?: number;
  success_probability?: number;
  samples: number;
  gini_impurity?: number;
  impurity?: number;
  depth: number;
  children?: TreeNode[];
  is_on_path?: boolean;
  decision_taken?: string;
  feature_value?: number;
  confidence?: number;
  is_beyond_max_depth?: boolean;
  is_left_child?: boolean;
}

interface EnhancedTreeViewerProps {
  treeId: number;
  treeData?: TreeNode;
  pathNodeIds?: Set<number>;
  showPathHighlighting?: boolean;
  onBack?: () => void;
  title?: string;
  subtitle?: string;
}

function EnhancedTreeNode({ 
  node, 
  pathNodeIds = new Set(), 
  showPathHighlighting = false 
}: { 
  node: TreeNode; 
  pathNodeIds?: Set<number>; 
  showPathHighlighting?: boolean; 
}) {
  const isLeaf = node.is_leaf;
  const isOnPath = showPathHighlighting && (node.is_on_path || pathNodeIds.has(node.node_id));
  const isBeyondMaxDepth = node.is_beyond_max_depth || false;
  
  const getNodeColor = () => {
    if (showPathHighlighting && isOnPath) {
      if (isBeyondMaxDepth) {
        return 'border-3 border-amber-500 bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/30 dark:to-amber-800/30 shadow-xl ring-2 ring-amber-300/50';
      }
      return 'border-3 border-blue-500 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 shadow-xl ring-2 ring-blue-300/50';
    }
    
    if (isLeaf) {
      const prediction = node.prediction || node.success_probability || 0;
      if (prediction >= 0.7) return 'border-2 border-green-500 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 shadow-lg';
      if (prediction >= 0.5) return 'border-2 border-yellow-500 bg-gradient-to-br from-yellow-50 to-yellow-100 dark:from-yellow-900/20 dark:to-yellow-800/20 shadow-lg';
      return 'border-2 border-red-500 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 shadow-lg';
    }
    
    return 'border-2 border-gray-300 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 hover:from-gray-100 hover:to-gray-200 dark:hover:from-gray-600 dark:hover:to-gray-700 shadow-md hover:shadow-lg';
  };

  const getNodeIcon = () => {
    if (isLeaf) {
      return React.createElement('div', {
        key: 'leaf-icon',
        className: 'w-6 h-6 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center'
      }, React.createElement('svg', {
        key: 'leaf-svg',
        className: 'w-4 h-4 text-green-600 dark:text-green-400',
        fill: 'currentColor',
        viewBox: '0 0 20 20'
      }, React.createElement('path', {
        key: 'leaf-path',
        fillRule: 'evenodd',
        d: 'M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z',
        clipRule: 'evenodd'
      })));
    }
    
    return React.createElement('div', {
      key: 'internal-icon',
      className: 'w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center'
    }, React.createElement('svg', {
      key: 'internal-svg',
      className: 'w-4 h-4 text-blue-600 dark:text-blue-400',
      fill: 'none',
      stroke: 'currentColor',
      viewBox: '0 0 24 24'
    }, React.createElement('path', {
      key: 'internal-path',
      strokeLinecap: 'round',
      strokeLinejoin: 'round',
      strokeWidth: 2,
      d: 'M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
    })));
  };

  return React.createElement('div', {
    className: 'flex flex-col items-center relative'
  }, [
    // Node
    React.createElement('div', {
      key: 'node',
      className: `relative rounded-lg p-3 min-w-[140px] max-w-[180px] cursor-pointer transition-all duration-300 hover:scale-105 transform ${getNodeColor()}`
    }, [
      // Path indicator
      showPathHighlighting && isOnPath ? React.createElement('div', {
        key: 'path-indicator',
        className: 'absolute -top-2 -right-2 w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg animate-pulse'
      }) : null,

      // Beyond max depth indicator
      isBeyondMaxDepth ? React.createElement('div', {
        key: 'depth-indicator',
        className: 'absolute -top-2 -left-2 w-4 h-4 bg-amber-500 rounded-full border-2 border-white shadow-lg'
      }, React.createElement('div', {
        className: 'absolute inset-0 rounded-full bg-amber-500 animate-ping opacity-75'
      })) : null,

      // Node Header
      React.createElement('div', {
        key: 'header',
        className: 'flex items-center gap-2 mb-2'
      }, [
        getNodeIcon(),
        React.createElement('div', { key: 'info' }, [
          React.createElement('div', {
            key: 'title',
            className: 'font-bold text-xs text-gray-900 dark:text-white'
          }, isLeaf ? 'Leaf' : `Node ${node.node_id}`),
          React.createElement('div', {
            key: 'depth',
            className: 'text-xs text-gray-500 dark:text-gray-400'
          }, `Depth ${node.depth}`)
        ])
      ]),

      // Decision indicator for path nodes (moved outside header)
      showPathHighlighting && isOnPath && node.decision_taken ? React.createElement('div', {
        key: 'decision-indicator',
        className: `mb-2 px-2 py-1 rounded-full text-xs font-bold text-center ${
          node.decision_taken === 'left' 
            ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' 
            : 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300'
        }`
      }, node.decision_taken === 'left' ? '← LEFT' : 'RIGHT →') : null,

      // Node Content
      !isLeaf ? React.createElement('div', {
        key: 'content',
        className: 'space-y-1 text-xs'
      }, [
        React.createElement('div', {
          key: 'feature',
          className: 'bg-white/50 dark:bg-gray-800/50 rounded p-2'
        }, [
          React.createElement('div', {
            key: 'feature-label',
            className: 'font-medium text-gray-700 dark:text-gray-300 mb-1'
          }, 'Feature:'),
          React.createElement('div', {
            key: 'feature-value',
            className: 'font-mono text-xs text-gray-900 dark:text-white break-all'
          }, node.feature_name)
        ]),
        React.createElement('div', {
          key: 'threshold',
          className: 'bg-white/50 dark:bg-gray-800/50 rounded p-2'
        }, [
          React.createElement('div', {
            key: 'threshold-label',
            className: 'font-medium text-gray-700 dark:text-gray-300 mb-1'
          }, 'Threshold:'),
          React.createElement('div', {
            key: 'threshold-value',
            className: 'font-mono text-xs text-gray-900 dark:text-white'
          }, `≤ ${node.threshold?.toFixed(3)}`)
        ]),
        
        // Input value for path nodes
        showPathHighlighting && isOnPath && node.feature_value !== undefined ? React.createElement('div', {
          key: 'input-value',
          className: 'bg-blue-50 dark:bg-blue-900/20 rounded p-2'
        }, [
          React.createElement('div', {
            key: 'input-label',
            className: 'font-medium text-blue-700 dark:text-blue-300 mb-1'
          }, 'Input Value:'),
          React.createElement('div', {
            key: 'input-val',
            className: 'font-mono text-xs text-blue-900 dark:text-blue-100'
          }, node.feature_value.toFixed(3)),
          React.createElement('div', {
            key: 'decision',
            className: 'text-xs text-blue-600 dark:text-blue-400 mt-1'
          }, `Decision: Go ${node.decision_taken}`)
        ]) : null
      ]) : React.createElement('div', {
        key: 'prediction',
        className: 'text-center'
      }, React.createElement('div', {
        className: 'bg-white/50 dark:bg-gray-800/50 rounded p-2'
      }, [
        React.createElement('div', {
          key: 'pred-label',
          className: 'text-xs font-medium text-gray-700 dark:text-gray-300 mb-1'
        }, 'Prediction'),
        React.createElement('div', {
          key: 'pred-value',
          className: `text-lg font-bold ${
            (node.prediction || node.success_probability || 0) >= 0.7 ? 'text-green-600 dark:text-green-400' :
            (node.prediction || node.success_probability || 0) >= 0.5 ? 'text-yellow-600 dark:text-yellow-400' :
            'text-red-600 dark:text-red-400'
          }`
        }, `${((node.prediction || node.success_probability || 0) * 100).toFixed(1)}%`),
        node.confidence !== undefined ? React.createElement('div', {
          key: 'confidence',
          className: 'text-xs text-gray-500 dark:text-gray-400 mt-1'
        }, `Confidence: ${(node.confidence * 100).toFixed(1)}%`) : null
      ])),

      // Node Statistics
      React.createElement('div', {
        key: 'stats',
        className: 'mt-2 pt-2 border-t border-gray-200 dark:border-gray-600'
      }, React.createElement('div', {
        className: 'grid grid-cols-2 gap-1 text-xs'
      }, [
        React.createElement('div', {
          key: 'samples',
          className: 'text-center'
        }, [
          React.createElement('div', {
            key: 'samples-value',
            className: 'font-bold text-gray-900 dark:text-white'
          }, node.samples),
          React.createElement('div', {
            key: 'samples-label',
            className: 'text-gray-500 dark:text-gray-400'
          }, 'Samples')
        ]),
        React.createElement('div', {
          key: 'gini',
          className: 'text-center'
        }, [
          React.createElement('div', {
            key: 'gini-value',
            className: 'font-bold text-gray-900 dark:text-white'
          }, (node.gini_impurity || node.impurity || 0).toFixed(3)),
          React.createElement('div', {
            key: 'gini-label',
            className: 'text-gray-500 dark:text-gray-400'
          }, 'Gini')
        ])
      ]))
    ]),

    // Children
    node.children && node.children.length > 0 ? React.createElement('div', {
      key: 'children',
      className: 'relative mt-8'
    }, [
      // Vertical line from parent
      React.createElement('div', {
        key: 'v-line',
        className: 'absolute top-0 left-1/2 transform -translate-x-1/2 w-0.5 h-6 bg-gray-400 dark:bg-gray-500'
      }),
      
      // Horizontal line connecting children
      node.children.length === 2 ? React.createElement('div', {
        key: 'h-line',
        className: 'absolute top-6 left-0 right-0 h-0.5 bg-gray-400 dark:bg-gray-500'
      }) : null,
      
      // Children container
      React.createElement('div', {
        key: 'children-container',
        className: 'flex justify-center items-start gap-12 pt-6'
      }, node.children.map((child, index) => 
        React.createElement('div', {
          key: child.node_id,
          className: 'flex flex-col items-center relative'
        }, [
          // Vertical line to child
          React.createElement('div', {
            key: 'child-v-line',
            className: 'w-0.5 h-6 bg-gray-400 dark:bg-gray-500 mb-2'
          }),
          
          // Branch label
          React.createElement('div', {
            key: 'branch-label',
            className: `mb-2 px-2 py-1 rounded text-xs font-bold ${
              (child.is_left_child !== undefined ? child.is_left_child : index === 0)
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' 
                : 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300'
            }`
          }, (child.is_left_child !== undefined ? child.is_left_child : index === 0) ? '≤ TRUE' : '> FALSE'),
          
          // Child node
          React.createElement(EnhancedTreeNode, {
            key: 'child-node',
            node: child,
            pathNodeIds,
            showPathHighlighting
          })
        ])
      ))
    ]) : null
  ]);
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

export default function EnhancedTreeViewer({ 
  treeId, 
  treeData, 
  pathNodeIds = new Set(), 
  showPathHighlighting = false,
  onBack,
  title,
  subtitle
}: EnhancedTreeViewerProps) {
  // Mock tree data for demonstration (same as SimpleTreeViewer)
  const mockTreeData: TreeNode = {
    node_id: 0,
    feature_name: "error_message_encoded",
    threshold: 2.5,
    is_leaf: false,
    samples: 1000,
    gini_impurity: 0.5,
    depth: 0,
    children: [
      {
        node_id: 1,
        feature_name: "billing_state_encoded",
        threshold: 15.5,
        is_leaf: false,
        samples: 600,
        gini_impurity: 0.4,
        depth: 1,
        is_left_child: true,
        children: [
          {
            node_id: 3,
            feature_name: "card_funding_encoded",
            threshold: 1.5,
            is_leaf: false,
            samples: 300,
            gini_impurity: 0.3,
            depth: 2,
            is_left_child: true,
            children: [
              {
                node_id: 7,
                is_leaf: true,
                prediction: 0.85,
                samples: 150,
                gini_impurity: 0.2,
                depth: 3,
                is_left_child: true
              },
              {
                node_id: 8,
                is_leaf: true,
                prediction: 0.65,
                samples: 150,
                gini_impurity: 0.4,
                depth: 3,
                is_left_child: false
              }
            ]
          },
          {
            node_id: 4,
            feature_name: "transaction_amount_encoded",
            threshold: 100.0,
            is_leaf: false,
            samples: 300,
            gini_impurity: 0.35,
            depth: 2,
            is_left_child: false,
            children: [
              {
                node_id: 9,
                is_leaf: true,
                prediction: 0.45,
                samples: 180,
                gini_impurity: 0.3,
                depth: 3,
                is_left_child: true
              },
              {
                node_id: 10,
                is_leaf: true,
                prediction: 0.25,
                samples: 120,
                gini_impurity: 0.25,
                depth: 3,
                is_left_child: false
              }
            ]
          }
        ]
      },
      {
        node_id: 2,
        feature_name: "card_network_encoded",
        threshold: 3.5,
        is_leaf: false,
        samples: 400,
        gini_impurity: 0.45,
        depth: 1,
        is_left_child: false,
        children: [
          {
            node_id: 5,
            feature_name: "merchant_category_encoded",
            threshold: 8.0,
            is_leaf: false,
            samples: 200,
            gini_impurity: 0.4,
            depth: 2,
            is_left_child: true,
            children: [
              {
                node_id: 11,
                is_leaf: true,
                prediction: 0.75,
                samples: 120,
                gini_impurity: 0.35,
                depth: 3,
                is_left_child: true
              },
              {
                node_id: 12,
                is_leaf: true,
                prediction: 0.55,
                samples: 80,
                gini_impurity: 0.45,
                depth: 3,
                is_left_child: false
              }
            ]
          },
          {
            node_id: 6,
            is_leaf: true,
            prediction: 0.15,
            samples: 200,
            gini_impurity: 0.25,
            depth: 2,
            is_left_child: false
          }
        ]
      }
    ]
  };

  const displayData = treeData || mockTreeData;
  const displayTitle = title || `Tree ${treeId} Structure`;
  const displaySubtitle = subtitle || 'Complete tree visualization showing all decision nodes and features';

  return React.createElement('div', {
    className: 'space-y-6'
  }, [
    // Header
    React.createElement('div', {
      key: 'header',
      className: 'bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6'
    }, [
      React.createElement('div', {
        key: 'header-content',
        className: 'flex items-center justify-between mb-4'
      }, [
        React.createElement('div', { key: 'title-section' }, [
          React.createElement('h2', {
            key: 'title',
            className: 'text-2xl font-bold text-gray-900 dark:text-white'
          }, displayTitle),
          React.createElement('p', {
            key: 'subtitle',
            className: 'text-gray-600 dark:text-gray-300'
          }, displaySubtitle)
        ]),
        
        onBack ? React.createElement('button', {
          key: 'back-button',
          onClick: onBack,
          className: 'bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-md transition-colors'
        }, 'Back') : null
      ]),

      // Tree Info
      React.createElement('div', {
        key: 'tree-info',
        className: 'grid grid-cols-2 md:grid-cols-4 gap-4 text-sm'
      }, [
        React.createElement('div', {
          key: 'tree-id',
          className: 'bg-gray-50 dark:bg-gray-700 rounded-lg p-3'
        }, [
          React.createElement('div', {
            key: 'tree-id-value',
            className: 'font-medium text-gray-900 dark:text-white'
          }, `Tree ${treeId}`),
          React.createElement('div', {
            key: 'tree-id-label',
            className: 'text-gray-500 dark:text-gray-400'
          }, 'Tree ID')
        ]),
        React.createElement('div', {
          key: 'samples',
          className: 'bg-gray-50 dark:bg-gray-700 rounded-lg p-3'
        }, [
          React.createElement('div', {
            key: 'samples-value',
            className: 'font-medium text-gray-900 dark:text-white'
          }, displayData.samples),
          React.createElement('div', {
            key: 'samples-label',
            className: 'text-gray-500 dark:text-gray-400'
          }, 'Total Samples')
        ]),
        React.createElement('div', {
          key: 'depth',
          className: 'bg-gray-50 dark:bg-gray-700 rounded-lg p-3'
        }, [
          React.createElement('div', {
            key: 'depth-value',
            className: 'font-medium text-gray-900 dark:text-white'
          }, getMaxDepth(displayData)),
          React.createElement('div', {
            key: 'depth-label',
            className: 'text-gray-500 dark:text-gray-400'
          }, 'Max Depth')
        ]),
        React.createElement('div', {
          key: 'nodes',
          className: 'bg-gray-50 dark:bg-gray-700 rounded-lg p-3'
        }, [
          React.createElement('div', {
            key: 'nodes-value',
            className: 'font-medium text-gray-900 dark:text-white'
          }, countNodes(displayData)),
          React.createElement('div', {
            key: 'nodes-label',
            className: 'text-gray-500 dark:text-gray-400'
          }, 'Total Nodes')
        ])
      ]),

      // Legend for path highlighting
      showPathHighlighting ? React.createElement('div', {
        key: 'legend',
        className: 'mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg'
      }, [
        React.createElement('h4', {
          key: 'legend-title',
          className: 'font-medium text-gray-900 dark:text-white mb-2'
        }, 'Legend:'),
        React.createElement('div', {
          key: 'legend-items',
          className: 'flex flex-wrap gap-4 text-sm'
        }, [
          React.createElement('div', {
            key: 'path-legend',
            className: 'flex items-center gap-2'
          }, [
            React.createElement('div', {
              key: 'path-legend-icon',
              className: 'w-4 h-4 border-2 border-blue-500 bg-blue-50 rounded'
            }),
            React.createElement('span', {
              key: 'path-legend-text',
              className: 'text-gray-700 dark:text-gray-300'
            }, 'Decision Path')
          ]),
          React.createElement('div', {
            key: 'green-legend',
            className: 'flex items-center gap-2'
          }, [
            React.createElement('div', {
              key: 'green-legend-icon',
              className: 'w-4 h-4 border border-green-500 bg-green-50 rounded'
            }),
            React.createElement('span', {
              key: 'green-legend-text',
              className: 'text-gray-700 dark:text-gray-300'
            }, 'High Prediction (≥70%)')
          ]),
          React.createElement('div', {
            key: 'yellow-legend',
            className: 'flex items-center gap-2'
          }, [
            React.createElement('div', {
              key: 'yellow-legend-icon',
              className: 'w-4 h-4 border border-yellow-500 bg-yellow-50 rounded'
            }),
            React.createElement('span', {
              key: 'yellow-legend-text',
              className: 'text-gray-700 dark:text-gray-300'
            }, 'Medium Prediction (50-70%)')
          ]),
          React.createElement('div', {
            key: 'red-legend',
            className: 'flex items-center gap-2'
          }, [
            React.createElement('div', {
              key: 'red-legend-icon',
              className: 'w-4 h-4 border border-red-500 bg-red-50 rounded'
            }),
            React.createElement('span', {
              key: 'red-legend-text',
              className: 'text-gray-700 dark:text-gray-300'
            }, 'Low Prediction (<50%)')
          ]),
          React.createElement('div', {
            key: 'amber-legend',
            className: 'flex items-center gap-2'
          }, [
            React.createElement('div', {
              key: 'amber-legend-icon',
              className: 'w-4 h-4 border border-amber-500 bg-amber-50 rounded'
            }),
            React.createElement('span', {
              key: 'amber-legend-text',
              className: 'text-gray-700 dark:text-gray-300'
            }, 'Beyond Max Depth')
          ])
        ])
      ]) : null
    ]),

    // Tree Visualization
    React.createElement('div', {
      key: 'visualization',
      className: 'bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6'
    }, [
      React.createElement('h3', {
        key: 'viz-title',
        className: 'text-xl font-semibold text-gray-900 dark:text-white mb-6'
      }, showPathHighlighting ? 'Tree Structure with Decision Path' : 'Tree Structure (Left to Right Flow)'),
      
      React.createElement('div', {
        key: 'viz-container',
        className: 'overflow-x-auto overflow-y-auto max-h-[80vh]'
      }, React.createElement('div', {
        className: 'min-w-max p-4'
      }, React.createElement(EnhancedTreeNode, {
        node: displayData,
        pathNodeIds,
        showPathHighlighting
      })))
    ])
  ]);
}
