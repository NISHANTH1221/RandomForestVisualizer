'use client';

import { motion } from 'framer-motion';
import TreeCard from '@/components/TreeCard';
import React, { useState, useEffect } from 'react';

export default function Home() {
  const [treeData, setTreeData] = React.useState<Array<{
    treeId: number;
    prediction: number;
    depth: number;
    nodeCount: number;
  }>>([]);

  // Generate consistent tree data on client side only
  React.useEffect(() => {
    const generateTreeData = () => {
      const data: Array<{
        treeId: number;
        prediction: number;
        depth: number;
        nodeCount: number;
      }> = [];
      for (let i = 0; i < 100; i++) {
        // Use a simple seeded random function for consistent results
        const seed = i + 1;
        const prediction = ((seed * 17) % 100);
        const depth = ((seed * 7) % 15) + 5;
        const nodeCount = ((seed * 23) % 200) + 50;
        
        data.push({
          treeId: i + 1,
          prediction,
          depth,
          nodeCount
        });
      }
      setTreeData(data);
    };

    generateTreeData();
  }, []);

  const handleTreeClick = (treeId: number) => {
    console.log(`Clicked tree ${treeId}`);
    // Navigate to tree viewer page
    window.location.href = `/tree-visualization?tree=${treeId}`;
  };

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.6,
        staggerChildren: 0.02
      }
    }
  };

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: { duration: 0.3 }
    }
  };

  const headerVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  };

  return React.createElement(motion.div, {
    className: 'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8',
    initial: 'hidden',
    animate: 'visible',
    variants: containerVariants
  }, [
    // Modern Header with Gradient Text
    React.createElement(motion.div, {
      key: 'header',
      className: 'text-center mb-12',
      variants: headerVariants
    }, [
      React.createElement('h1', {
        key: 'title',
        className: 'text-display-lg gradient-text mb-4'
      }, 'Random Forest Trees'),
      React.createElement('p', {
        key: 'subtitle',
        className: 'text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto leading-relaxed'
      }, 'Explore the intricate structure of 100 decision trees working together in harmony')
    ]),

    // Tree Grid with Modern Glass Effect
    React.createElement(motion.div, {
      key: 'tree-grid',
      className: 'card-modern rounded-2xl p-8 shadow-2xl',
      variants: headerVariants
    }, [
      React.createElement('div', {
        key: 'grid-header',
        className: 'flex items-center justify-between mb-6'
      }, [
        React.createElement('div', { key: 'grid-title' }, [
          React.createElement('h2', {
            key: 'title',
            className: 'text-xl font-semibold text-gray-900 dark:text-white'
          }, '100 Decision Trees'),
          React.createElement('p', {
            key: 'description',
            className: 'text-gray-600 dark:text-gray-300'
          }, 'Each tree shows a different perspective of the Random Forest model')
        ]),
        React.createElement('div', {
          key: 'count',
          className: 'text-sm text-gray-500 dark:text-gray-400'
        }, 'Total: 100 trees')
      ]),
      
      // Tree grid with TreeCard components
      React.createElement(motion.div, {
        key: 'grid',
        className: 'grid grid-cols-10 gap-3',
        variants: containerVariants
      }, treeData.map((tree, i) =>
        React.createElement(motion.div, {
          key: tree.treeId,
          variants: cardVariants
        }, React.createElement(TreeCard, {
          treeId: tree.treeId,
          prediction: tree.prediction,
          depth: tree.depth,
          nodeCount: tree.nodeCount,
          onClick: handleTreeClick
        }))
      )),

      // Modern Legend with Glass Effect
      React.createElement('div', {
        key: 'legend',
        className: 'mt-8 p-6 glass rounded-xl border border-white/20'
      }, [
        React.createElement('h3', {
          key: 'legend-title',
          className: 'text-lg font-semibold text-gray-900 dark:text-white mb-4 gradient-text-cool'
        }, 'Performance Legend'),
        React.createElement('div', {
          key: 'legend-items',
          className: 'flex flex-wrap gap-6 text-sm'
        }, [
          React.createElement('div', {
            key: 'green',
            className: 'flex items-center gap-3 px-3 py-2 rounded-lg bg-green-50 dark:bg-green-900/20'
          }, [
            React.createElement('div', {
              key: 'color',
              className: 'w-5 h-5 bg-gradient-to-r from-green-400 to-green-600 rounded-full shadow-lg'
            }),
            React.createElement('span', {
              key: 'label',
              className: 'font-medium text-green-700 dark:text-green-300'
            }, 'High Accuracy (≥70%)')
          ]),
          React.createElement('div', {
            key: 'yellow',
            className: 'flex items-center gap-3 px-3 py-2 rounded-lg bg-yellow-50 dark:bg-yellow-900/20'
          }, [
            React.createElement('div', {
              key: 'color',
              className: 'w-5 h-5 bg-gradient-to-r from-yellow-400 to-yellow-600 rounded-full shadow-lg'
            }),
            React.createElement('span', {
              key: 'label',
              className: 'font-medium text-yellow-700 dark:text-yellow-300'
            }, 'Medium Accuracy (50-70%)')
          ]),
          React.createElement('div', {
            key: 'red',
            className: 'flex items-center gap-3 px-3 py-2 rounded-lg bg-red-50 dark:bg-red-900/20'
          }, [
            React.createElement('div', {
              key: 'color',
              className: 'w-5 h-5 bg-gradient-to-r from-red-400 to-red-600 rounded-full shadow-lg'
            }),
            React.createElement('span', {
              key: 'label',
              className: 'font-medium text-red-700 dark:text-red-300'
            }, 'Low Accuracy (<50%)')
          ])
        ])
      ])
    ])
  ]);
}
