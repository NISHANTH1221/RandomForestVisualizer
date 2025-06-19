const Footer = () => {
  return (
    <footer className="glass-card border-t border-white/10 mt-16 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-6 md:space-y-0">
          <div className="text-center md:text-left">
            <p className="text-lg font-semibold gradient-text mb-2">
              Random Forest Visualization Platform
            </p>
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              Built with Next.js, TypeScript & Tailwind CSS
            </p>
          </div>
          
          <div className="flex items-center space-x-8">
            <div className="text-center p-4 glass rounded-xl">
              <p className="text-3xl font-bold gradient-text-cool">100</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Trees</p>
            </div>
            <div className="text-center p-4 glass rounded-xl">
              <p className="text-3xl font-bold gradient-text-warm">5</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Features</p>
            </div>
            <div className="text-center p-4 glass rounded-xl">
              <p className="text-3xl font-bold gradient-text">ML</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Model</p>
            </div>
          </div>
        </div>
        
        <div className="mt-8 pt-6 border-t border-white/10">
          <p className="text-center text-gray-500 dark:text-gray-400 text-sm">
            © 2024 Random Forest Visualization. Interactive machine learning model exploration.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
