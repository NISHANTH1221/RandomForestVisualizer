import DecisionPath from '@/components/DecisionPath';

export default function DecisionPathPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Decision Path Visualization
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Trace the exact path a transaction takes through a specific tree in the Random Forest model.
        </p>
      </div>

      <DecisionPath />
    </div>
  );
}
