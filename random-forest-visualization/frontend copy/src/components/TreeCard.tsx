'use client';

import { motion } from 'framer-motion';

interface TreeCardProps {
  treeId: number;
  prediction?: number;
  depth?: number;
  nodeCount?: number;
  isLoading?: boolean;
  onClick?: (treeId: number) => void;
}

export default function TreeCard({ 
  treeId, 
  prediction, 
  depth, 
  nodeCount, 
  isLoading = false, 
  onClick 
}: TreeCardProps) {

  // Determine color based on prediction percentage with modern gradients
  const getColorClasses = () => {
    if (isLoading || prediction === undefined) {
      return 'from-gray-400 via-gray-500 to-gray-600';
    }
    
    if (prediction >= 70) {
      return 'from-emerald-400 via-green-500 to-teal-600';
    } else if (prediction >= 50) {
      return 'from-amber-400 via-yellow-500 to-orange-600';
    } else {
      return 'from-rose-400 via-red-500 to-pink-600';
    }
  };

  // Get success indicator
  const getSuccessIndicator = () => {
    if (isLoading || prediction === undefined) return null;
    
    if (prediction >= 70) {
      return (
        <div className="absolute top-1 right-1 w-2 h-2 bg-green-500 rounded-full shadow-sm"></div>
      );
    } else if (prediction >= 50) {
      return (
        <div className="absolute top-1 right-1 w-2 h-2 bg-yellow-500 rounded-full shadow-sm"></div>
      );
    } else {
      return (
        <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full shadow-sm"></div>
      );
    }
  };

  const handleClick = () => {
    if (!isLoading && onClick) {
      onClick(treeId);
    }
  };

  return (
    <motion.div
      className={`
        relative aspect-square rounded-xl cursor-pointer overflow-hidden
        ${isLoading ? 'animate-pulse' : ''}
        bg-gradient-to-br ${getColorClasses()}
        shadow-lg hover:shadow-2xl transition-shadow duration-300
        border border-white/20
      `}
      onClick={handleClick}
      whileHover={{ 
        scale: 1.05, 
        zIndex: 10,
        boxShadow: "0 20px 40px rgba(0,0,0,0.3), 0 0 20px rgba(255,255,255,0.1)"
      }}
      whileTap={{ scale: 0.95 }}
      transition={{ 
        type: "spring", 
        stiffness: 400, 
        damping: 25 
      }}
    >
      {/* Success indicator */}
      {getSuccessIndicator()}
      
      {/* Tree ID */}
      <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
        <motion.div 
          className="text-xs font-bold mb-1"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          {isLoading ? '...' : treeId}
        </motion.div>
        
        {/* Prediction percentage */}
        {!isLoading && prediction !== undefined && (
          <motion.div 
            className="text-[10px] font-medium opacity-90"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            {prediction.toFixed(1)}%
          </motion.div>
        )}
      </div>

      {/* Hover overlay with additional info */}
      <motion.div 
        className="absolute inset-0 bg-black/80 rounded-lg flex flex-col items-center justify-center text-white text-[10px]"
        initial={{ opacity: 0 }}
        whileHover={{ opacity: 1 }}
        transition={{ duration: 0.2 }}
      >
        <div className="font-bold mb-1">Tree {treeId}</div>
        {prediction !== undefined && (
          <div className="mb-1">{prediction.toFixed(1)}% Success</div>
        )}
        {depth !== undefined && (
          <div className="mb-1">Depth: {depth}</div>
        )}
        {nodeCount !== undefined && (
          <div>Nodes: {nodeCount}</div>
        )}
        <div className="mt-1 text-[8px] opacity-75">Click to explore</div>
      </motion.div>

      {/* Loading state */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div 
            className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
        </div>
      )}
    </motion.div>
  );
}
