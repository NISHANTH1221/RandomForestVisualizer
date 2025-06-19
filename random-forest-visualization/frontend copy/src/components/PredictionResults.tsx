'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { PredictionResponse, IndividualPrediction } from '@/services/api';

interface TreePredictionCardProps {
  prediction: IndividualPrediction;
  index: number;
  isSelected: boolean;
  onClick: () => void;
}

function TreePredictionCardSkeleton({ index }: { index: number }) {
  return (
    <div
      className="p-4 rounded-xl border-2 border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
      style={{
        animationDelay: `${index * 50}ms`,
        animation: 'fadeInUp 0.6s ease-out forwards'
      }}
    >
      {/* Icon skeleton */}
      <div className="flex items-center justify-center mb-2">
        <div className="w-6 h-6 bg-gray-300 dark:bg-gray-600 rounded-full skeleton"></div>
      </div>

      {/* Tree ID skeleton */}
      <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded mb-1 skeleton"></div>

      {/* Percentage skeleton */}
      <div className="h-5 bg-gray-300 dark:bg-gray-600 rounded mb-1 skeleton"></div>

      {/* Progress bar skeleton */}
      <div className="w-full h-1.5 bg-gray-300 dark:bg-gray-600 rounded mb-2 skeleton"></div>
    </div>
  );
}

function TreePredictionCard({ prediction, index, isSelected, onClick }: TreePredictionCardProps) {
  const getCardTheme = (pred: number) => {
    if (pred >= 0.7) {
      return {
        gradient: 'from-emerald-50 via-green-50 to-teal-50 dark:from-emerald-950 dark:via-green-950 dark:to-teal-950',
        border: 'border-emerald-200 dark:border-emerald-700',
        accent: 'bg-emerald-500',
        textColor: 'text-emerald-700 dark:text-emerald-300',
        icon: '✓',
        iconBg: 'bg-emerald-100 dark:bg-emerald-900',
        shadow: 'shadow-emerald-200/50 dark:shadow-emerald-900/50'
      };
    } else if (pred >= 0.5) {
      return {
        gradient: 'from-amber-50 via-yellow-50 to-orange-50 dark:from-amber-950 dark:via-yellow-950 dark:to-orange-950',
        border: 'border-amber-200 dark:border-amber-700',
        accent: 'bg-amber-500',
        textColor: 'text-amber-700 dark:text-amber-300',
        icon: '⚠',
        iconBg: 'bg-amber-100 dark:bg-amber-900',
        shadow: 'shadow-amber-200/50 dark:shadow-amber-900/50'
      };
    } else {
      return {
        gradient: 'from-red-50 via-rose-50 to-pink-50 dark:from-red-950 dark:via-rose-950 dark:to-pink-950',
        border: 'border-red-200 dark:border-red-700',
        accent: 'bg-red-500',
        textColor: 'text-red-700 dark:text-red-300',
        icon: '✗',
        iconBg: 'bg-red-100 dark:bg-red-900',
        shadow: 'shadow-red-200/50 dark:shadow-red-900/50'
      };
    }
  };

  const theme = getCardTheme(prediction.prediction.value);
  const percentage = (prediction.prediction.value * 100).toFixed(1);

  return (
    <button
      onClick={onClick}
      className={`
        group relative w-full aspect-square rounded-xl border-2 transition-all duration-300 ease-out
        transform hover:scale-105 hover:shadow-lg hover:-translate-y-1
        focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:ring-offset-2
        bg-gradient-to-br ${theme.gradient}
        ${theme.border}
        ${isSelected ? 'ring-2 ring-blue-500 dark:ring-blue-400 scale-105 shadow-lg -translate-y-1' : ''}
        hover:${theme.shadow}
        overflow-hidden
      `}
      style={{
        animationDelay: `${index * 20}ms`,
        animation: 'fadeInUp 0.6s ease-out forwards'
      }}
    >
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5">
        <svg className="w-full h-full" viewBox="0 0 100 100">
          <defs>
            <pattern id={`pattern-${prediction.tree_id}`} x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
              <circle cx="10" cy="10" r="1" fill="currentColor" />
            </pattern>
          </defs>
          <rect width="100" height="100" fill={`url(#pattern-${prediction.tree_id})`} />
        </svg>
      </div>

      {/* Header with tree ID and status icon */}
      <div className="absolute top-2 left-2 right-2 flex justify-between items-center z-10">
        <div className={`px-2 py-1 rounded-md text-xs font-bold ${theme.iconBg} ${theme.textColor}`}>
          #{prediction.tree_id}
        </div>
        <div className={`w-6 h-6 rounded-full ${theme.iconBg} flex items-center justify-center text-xs font-bold ${theme.textColor}`}>
          {theme.icon}
        </div>
      </div>

      {/* Main content area */}
      <div className="absolute inset-0 flex flex-col items-center justify-center p-4">
        {/* Large percentage display */}
        <div className={`text-3xl font-black mb-2 ${theme.textColor}`}>
          {percentage}%
        </div>

        {/* Progress bar */}
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-3 overflow-hidden">
          <div 
            className={`h-full ${theme.accent} transition-all duration-1000 ease-out rounded-full`}
            style={{ 
              width: `${prediction.prediction.value * 100}%`,
              animationDelay: `${index * 50 + 300}ms`
            }}
          />
        </div>

        {/* Tree visualization */}
        <div className={`w-8 h-8 rounded-lg ${theme.iconBg} flex items-center justify-center mb-2`}>
          <svg className={`w-5 h-5 ${theme.textColor}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
          </svg>
        </div>

        {/* Confidence indicator */}
        <div className={`text-xs font-medium ${theme.textColor} opacity-75`}>
          {prediction.prediction.confidence ? `${(prediction.prediction.confidence * 100).toFixed(0)}% conf` : 'High conf'}
        </div>
      </div>

      {/* Bottom accent line */}
      <div className={`absolute bottom-0 left-0 right-0 h-1 ${theme.accent}`} />

      {/* Hover overlay */}
      <div className="absolute inset-0 bg-white/10 dark:bg-black/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl" />

      {/* Click feedback */}
      <div className="absolute inset-0 bg-white/20 dark:bg-black/20 scale-0 group-active:scale-100 transition-transform duration-150 rounded-xl" />

      {/* Tooltip */}
      <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none z-20">
        <div className="bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-xs px-2 py-1 rounded shadow-lg whitespace-nowrap">
          Tree {prediction.tree_id} • {percentage}%
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-2 border-r-2 border-t-2 border-transparent border-t-gray-900 dark:border-t-gray-100" />
        </div>
      </div>
    </button>
  );
}

interface PredictionResultsProps {
  predictions: PredictionResponse | null;
  formData: any;
  onTreeClick?: (treeId: number) => void;
}

export default function PredictionResults({ predictions, formData, onTreeClick }: PredictionResultsProps) {
  const router = useRouter();
  const [selectedTreeId, setSelectedTreeId] = useState<number | null>(null);

  const handleTreeClick = async (treeId: number) => {
    if (!predictions) return;
    
    setSelectedTreeId(treeId);
    try {
      // Store prediction data and navigate to decision path
      sessionStorage.setItem('currentPrediction', JSON.stringify({
        formData,
        treeId,
        prediction: predictions.individual_predictions.find(p => p.tree_id === treeId)
      }));
      
      if (onTreeClick) {
        onTreeClick(treeId);
      } else {
        router.push(`/decision-path?tree=${treeId}`);
      }
    } catch (error) {
      console.error('Failed to navigate to decision path:', error);
    }
  };

  const handleTreeVisualization = async (treeId: number) => {
    if (!predictions) return;
    
    try {
      // Store prediction data and navigate to tree visualization
      sessionStorage.setItem('currentPrediction', JSON.stringify({
        formData,
        treeId,
        prediction: predictions.individual_predictions.find(p => p.tree_id === treeId)
      }));
      
      router.push(`/tree-visualization?tree=${treeId}`);
    } catch (error) {
      console.error('Failed to navigate to tree visualization:', error);
    }
  };

  const getSuccessColor = (prediction: number): string => {
    if (prediction >= 0.7) return 'text-green-600 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-900/20 dark:border-green-800';
    if (prediction >= 0.5) return 'text-yellow-600 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-900/20 dark:border-yellow-800';
    return 'text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-900/20 dark:border-red-800';
  };

  const getSuccessIcon = (prediction: number) => {
    if (prediction >= 0.7) {
      return (
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      );
    }
    if (prediction >= 0.5) {
      return (
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
      );
    }
    return (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
    );
  };

  if (!predictions) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
          Prediction Results
        </h2>
        
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No Predictions Yet
          </h3>
          <p className="text-gray-500 dark:text-gray-400 mb-6">
            Fill out the form and submit to see predictions from all 100 trees
          </p>
          
          <div className="space-y-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <div className="text-sm text-gray-600 dark:text-gray-300">
                Ensemble Prediction: <span className="font-medium">--</span>
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <div className="text-sm text-gray-600 dark:text-gray-300">
                Individual Trees: <span className="font-medium">0/100</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
        Prediction Results
      </h2>
      
      <div className="space-y-6">
        {/* Enhanced Ensemble Result with Circular Progress */}
        <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-blue-900/20 dark:via-indigo-900/20 dark:to-purple-900/20 rounded-xl p-8 border border-blue-200 dark:border-blue-800 shadow-lg">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 text-center">
            Overall Success Rate
          </h3>
          
          <div className="flex flex-col lg:flex-row items-center justify-center gap-8">
            {/* Circular Progress Indicator */}
            <div className="relative flex items-center justify-center">
              <svg className="w-48 h-48 transform -rotate-90" viewBox="0 0 200 200">
                {/* Background Circle */}
                <circle
                  cx="100"
                  cy="100"
                  r="85"
                  stroke="currentColor"
                  strokeWidth="12"
                  fill="none"
                  className="text-gray-200 dark:text-gray-700"
                />
                {/* Progress Circle */}
                <circle
                  cx="100"
                  cy="100"
                  r="85"
                  stroke="url(#gradient)"
                  strokeWidth="12"
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray={`${2 * Math.PI * 85}`}
                  strokeDashoffset={`${2 * Math.PI * 85 * (1 - predictions.ensemble_prediction.value)}`}
                  className="transition-all duration-2000 ease-out"
                  style={{
                    animation: 'drawCircle 2s ease-out forwards'
                  }}
                />
                {/* Gradient Definition */}
                <defs>
                  <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#3B82F6" />
                    <stop offset="50%" stopColor="#6366F1" />
                    <stop offset="100%" stopColor="#8B5CF6" />
                  </linearGradient>
                </defs>
              </svg>
              
              {/* Center Content */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <div className="text-5xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  {(predictions.ensemble_prediction.value * 100).toFixed(1)}%
                </div>
                <div className="text-sm font-medium text-gray-600 dark:text-gray-300 mt-1">
                  Success Rate
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {predictions.statistics.total_trees} Trees
                </div>
              </div>
            </div>
            
            {/* Statistics and Confidence */}
            <div className="flex flex-col space-y-6 text-center lg:text-left">
              {/* Confidence Interval */}
              <div className="bg-white/60 dark:bg-gray-800/60 rounded-lg p-4 backdrop-blur-sm">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                  Confidence Interval
                </h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-300">Lower Bound:</span>
                    <span className="font-bold text-blue-600 dark:text-blue-400">
                      {predictions.statistics.confidence_interval?.lower 
                        ? (predictions.statistics.confidence_interval.lower * 100).toFixed(1)
                        : 'N/A'}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-300">Upper Bound:</span>
                    <span className="font-bold text-blue-600 dark:text-blue-400">
                      {predictions.statistics.confidence_interval?.upper 
                        ? (predictions.statistics.confidence_interval.upper * 100).toFixed(1)
                        : 'N/A'}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-300">Confidence:</span>
                    <span className="font-bold text-green-600 dark:text-green-400">
                      {(predictions.ensemble_prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Comparison with Individual Trees */}
              <div className="bg-white/60 dark:bg-gray-800/60 rounded-lg p-4 backdrop-blur-sm">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                  Tree Agreement
                </h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-300">Average Tree:</span>
                    <span className="font-bold text-gray-700 dark:text-gray-300">
                      {(predictions.statistics.average_prediction * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-300">Std Deviation:</span>
                    <span className="font-bold text-gray-700 dark:text-gray-300">
                      ±{(predictions.statistics.prediction_std * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-300">High Confidence:</span>
                    <span className="font-bold text-green-600 dark:text-green-400">
                      {predictions.individual_predictions.filter(p => p.prediction.value >= 0.7).length} trees
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Success Indicator */}
              <div className={`
                rounded-lg p-4 text-center font-bold text-lg
                ${predictions.ensemble_prediction.value >= 0.7 
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 border border-green-300 dark:border-green-700' 
                  : predictions.ensemble_prediction.value >= 0.5
                  ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 border border-yellow-300 dark:border-yellow-700'
                  : 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300 border border-red-300 dark:border-red-700'
                }
              `}>
                {predictions.ensemble_prediction.value >= 0.7 
                  ? '✅ High Success Probability' 
                  : predictions.ensemble_prediction.value >= 0.5
                  ? '⚠️ Moderate Success Probability'
                  : '❌ Low Success Probability'
                }
              </div>
            </div>
          </div>
        </div>


        {/* Enhanced Statistics */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 dark:text-white mb-3">
            Prediction Statistics
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-300">Average:</span>
              <span className="font-medium">
                {(predictions.statistics.average_prediction * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-300">Std Dev:</span>
              <span className="font-medium">
                {(predictions.statistics.prediction_std * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-300">CI Lower:</span>
              <span className="font-medium">
                {predictions.statistics.confidence_interval?.lower 
                  ? (predictions.statistics.confidence_interval.lower * 100).toFixed(1)
                  : 'N/A'}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-300">CI Upper:</span>
              <span className="font-medium">
                {predictions.statistics.confidence_interval?.upper 
                  ? (predictions.statistics.confidence_interval.upper * 100).toFixed(1)
                  : 'N/A'}%
              </span>
            </div>
          </div>
          
          {/* Success Rate Distribution */}
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
            <div className="text-xs text-gray-600 dark:text-gray-300 mb-2">Success Rate Distribution:</div>
            <div className="flex gap-2 text-xs">
              <div className="flex items-center gap-1">
                <span className="text-green-600 dark:text-green-400 font-medium">
                  {predictions.individual_predictions.filter(p => p.prediction.value >= 0.7).length}
                </span>
                <span className="text-gray-500">high</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-yellow-600 dark:text-yellow-400 font-medium">
                  {predictions.individual_predictions.filter(p => p.prediction.value >= 0.5 && p.prediction.value < 0.7).length}
                </span>
                <span className="text-gray-500">medium</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-red-600 dark:text-red-400 font-medium">
                  {predictions.individual_predictions.filter(p => p.prediction.value < 0.5).length}
                </span>
                <span className="text-gray-500">low</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
