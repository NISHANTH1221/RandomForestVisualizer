'use client';

import React, { useState, useEffect, Fragment } from 'react';
import { useRouter } from 'next/navigation';
import TreeCard from '@/components/TreeCard';
import { getAllTrees, TreeMetadata, TreesResponse } from '@/services/api';

export default function TreesPage() {
  const router = useRouter();
  const [treesData, setTreesData] = useState<TreesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTreeId, setSelectedTreeId] = useState<number | null>(null);

  // Load trees data on component mount
  useEffect(() => {
    loadTreesData();
  }, []);

  const loadTreesData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await getAllTrees();
      setTreesData(data);
      
      console.log('Trees data loaded successfully:', {
        totalTrees: data.trees.length,
        successfulExtractions: data.forest_statistics.successful_extractions,
        failedExtractions: data.forest_statistics.failed_extractions
      });
      
    } catch (err) {
      console.error('Failed to load trees data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load trees data');
    } finally {
      setLoading(false);
    }
  };

  const handleTreeClick = (treeId: number) => {
    setSelectedTreeId(treeId);
    // Navigate to tree detail view or show modal
    console.log(`Tree ${treeId} clicked`);
    // For now, we'll just log it. Later we can navigate to a detail page
    // router.push(`/trees/${treeId}`);
  };

  const handleRefresh = () => {
    loadTreesData();
  };

  // Calculate grid layout based on screen size
  const getGridCols = () => {
    // 10x10 for desktop (xl), responsive for smaller screens
    return 'grid-cols-2 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-8 xl:grid-cols-10';
  };

  // Generate grid position info for numbering
  const getGridPosition = (index: number) => {
    const cols = 10; // Desktop columns
    const row = Math.floor(index / cols) + 1;
    const col = (index % cols) + 1;
    return { row, col, position: index + 1 };
  };

  // Generate mock prediction for trees (since we don't have predictions yet)
  const getMockPrediction = (treeId: number): number => {
    // Generate consistent mock predictions based on tree ID
    const seed = treeId * 17 + 42; // Simple pseudo-random
    return 30 + (seed % 60); // Range: 30-90%
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Tree Grid View
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Explore all {treesData?.forest_statistics.total_trees || 100} trees in your Random Forest model. 
          Click on any tree to view its detailed structure.
        </p>
      </div>

      {/* Error State */}
      {error && (
        <div className="mb-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800 dark:text-red-200">
                  Failed to load trees data
                </h3>
                <div className="mt-2 text-sm text-red-700 dark:text-red-300">
                  {error}
                </div>
              </div>
            </div>
            <button
              onClick={handleRefresh}
              className="ml-3 bg-red-100 dark:bg-red-800 text-red-800 dark:text-red-200 px-3 py-1 rounded-md text-sm font-medium hover:bg-red-200 dark:hover:bg-red-700 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Tree Grid */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-6 space-y-4 sm:space-y-0">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Random Forest Trees
              {treesData && (
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                  ({treesData.forest_statistics.successful_extractions}/{treesData.forest_statistics.total_trees} loaded)
                </span>
              )}
            </h2>
            {treesData && (
              <div className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Avg Depth: {treesData.forest_statistics.average_depth.toFixed(1)} | 
                Avg Nodes: {treesData.forest_statistics.average_node_count.toFixed(0)}
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Grid: Responsive Layout
            </div>
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="px-3 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-md hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <div className="text-gray-600 dark:text-gray-300">Loading trees data...</div>
            </div>
          </div>
        )}

        {/* Tree Grid Container with Scroll */}
        {!loading && treesData && (
          <React.Fragment>
            {/* Grid Info */}
            <div className="mb-4 flex flex-col sm:flex-row sm:justify-between sm:items-center space-y-2 sm:space-y-0">
              <div className="text-sm text-gray-600 dark:text-gray-400">
                <span className="font-medium">Grid Layout:</span> 
                <span className="ml-1">
                  <span className="hidden xl:inline">10×10 (Desktop)</span>
                  <span className="hidden lg:inline xl:hidden">8×13 (Large)</span>
                  <span className="hidden md:inline lg:hidden">5×20 (Medium)</span>
                  <span className="hidden sm:inline md:hidden">4×25 (Small)</span>
                  <span className="inline sm:hidden">2×50 (Mobile)</span>
                </span>
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Scroll to view all trees
              </div>
            </div>

            {/* Scrollable Grid Container */}
            <div className="relative">
              {/* Grid with fixed height and scroll */}
              <div className="max-h-[600px] overflow-y-auto overflow-x-hidden scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 scrollbar-track-transparent">
                <div className={`grid ${getGridCols()} gap-3 p-1`}>
                  {treesData.trees.map((tree: TreeMetadata, index: number) => {
                    const gridPos = getGridPosition(index);
                    return (
                      <div key={tree.tree_id} className="relative group">
                        {/* Grid Position Indicator (visible on hover) */}
                        <div className="absolute -top-2 -left-2 z-20 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                          <div className="bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-xs px-2 py-1 rounded-md shadow-lg">
                            #{gridPos.position} ({gridPos.row},{gridPos.col})
                          </div>
                        </div>
                        
                        <TreeCard
                          treeId={tree.tree_id}
                          prediction={getMockPrediction(tree.tree_id)}
                          depth={tree.depth}
                          nodeCount={tree.node_count}
                          isLoading={false}
                          onClick={handleTreeClick}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Scroll Indicators */}
              <div className="absolute top-0 right-0 bg-gradient-to-l from-white dark:from-gray-800 to-transparent w-8 h-full pointer-events-none opacity-50"></div>
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white dark:from-gray-800 to-transparent h-8 pointer-events-none opacity-50"></div>
            </div>

            {/* Grid Navigation Helper */}
            <div className="mt-4 flex justify-center">
              <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Hover for grid position</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                  <span>Scroll to explore all trees</span>
                </div>
              </div>
            </div>

            {/* Grid Footer */}
            <div className="mt-6 flex flex-col sm:flex-row sm:justify-between sm:items-center space-y-2 sm:space-y-0">
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Click on any tree to view its structure and make predictions
              </div>
              
              {/* Color Legend */}
              <div className="flex items-center space-x-4 text-xs">
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-gradient-to-br from-green-400 to-green-600 rounded-full"></div>
                  <span className="text-gray-600 dark:text-gray-300">High (≥70%)</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-full"></div>
                  <span className="text-gray-600 dark:text-gray-300">Medium (50-70%)</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-gradient-to-br from-red-400 to-red-600 rounded-full"></div>
                  <span className="text-gray-600 dark:text-gray-300">Low (&lt;50%)</span>
                </div>
              </div>
            </div>
          </React.Fragment>
        )}

        {/* Empty State */}
        {!loading && !treesData && !error && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="text-gray-400 dark:text-gray-500 mb-2">
                <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">No trees data</h3>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Unable to load Random Forest trees data.
              </p>
              <button
                onClick={handleRefresh}
                className="mt-3 px-4 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
              >
                Try Again
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Debug Info (only in development) */}
      {process.env.NODE_ENV === 'development' && treesData && (
        <div className="mt-6 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">Debug Info</h3>
          <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
            <div>Model Type: {treesData.model_info.model_type}</div>
            <div>Features: {treesData.model_info.n_features}</div>
            <div>Extraction Time: {treesData.extraction_info.timestamp}</div>
            <div>API Version: {treesData.extraction_info.api_version}</div>
            <div>Selected Tree: {selectedTreeId !== null ? selectedTreeId : 'None'}</div>
          </div>
        </div>
      )}
    </div>
  );
}
