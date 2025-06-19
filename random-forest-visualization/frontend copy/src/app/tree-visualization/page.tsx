'use client';

import * as React from 'react';
import { useSearchParams } from 'next/navigation';
import SimpleTreeViewer from '@/components/SimpleTreeViewer';

export default function TreeVisualizationPage() {
  const [treeId, setTreeId] = React.useState(1);

  React.useEffect(() => {
    // Get tree ID from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const treeParam = urlParams.get('tree');
    if (treeParam) {
      setTreeId(parseInt(treeParam));
    }
  }, []);

  const handleBack = () => {
    window.location.href = '/';
  };

  return React.createElement('div', {
    className: 'min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-gray-900 dark:via-slate-900 dark:to-indigo-950 py-8',
    style: {
      backgroundImage: `
        radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%)
      `
    }
  }, React.createElement('div', {
    className: 'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'
  }, React.createElement(SimpleTreeViewer, {
    treeId: treeId,
    onBack: handleBack
  })));
}
