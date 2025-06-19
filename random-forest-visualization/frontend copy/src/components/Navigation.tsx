'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import * as React from 'react';

const Navigation = () => {
  const pathname = usePathname();
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  const navItems = [
    { href: '/', label: 'Tree Grid', icon: '🌳' },
    { href: '/predict', label: 'Predictions', icon: '🎯' },
    { href: '/decision-path', label: 'Decision Path', icon: '🛤️' },
  ];

  return React.createElement('nav', {
    className: 'glass-card border-b border-white/10 backdrop-blur-xl'
  }, React.createElement('div', {
    className: 'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'
  }, React.createElement('div', {
    className: 'flex justify-between items-center h-16'
  }, [
    // Logo and Title
    React.createElement('div', {
      key: 'logo',
      className: 'flex items-center'
    }, React.createElement(Link, {
      href: '/',
      className: 'flex items-center space-x-3'
    }, [
      React.createElement('div', {
        key: 'logo-icon',
        className: 'w-10 h-10 bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg'
      }, React.createElement('span', {
        className: 'text-white font-bold text-lg'
      }, 'RF')),
      React.createElement('div', {
        key: 'logo-text'
      }, [
        React.createElement('h1', {
          key: 'title',
          className: 'text-xl font-bold gradient-text'
        }, 'Random Forest Visualization'),
        React.createElement('p', {
          key: 'subtitle',
          className: 'text-xs text-gray-500 dark:text-gray-400 font-medium'
        }, 'Interactive ML Model Explorer')
      ])
    ])),

    // Desktop Navigation
    React.createElement('div', {
      key: 'desktop-nav',
      className: 'hidden md:flex items-center space-x-1'
    }, navItems.map((item) => 
      React.createElement(Link, {
        key: item.href,
        href: item.href,
        className: `px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center space-x-2 ${
          pathname === item.href
            ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
            : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white'
        }`
      }, [
        React.createElement('span', { key: 'icon' }, item.icon),
        React.createElement('span', { key: 'label' }, item.label)
      ])
    )),

    // Mobile menu button
    React.createElement('div', {
      key: 'mobile-button',
      className: 'md:hidden'
    }, React.createElement('button', {
      onClick: () => setIsMenuOpen(!isMenuOpen),
      className: 'p-2 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500',
      'aria-label': 'Toggle menu'
    }, React.createElement('svg', {
      className: 'w-6 h-6',
      fill: 'none',
      stroke: 'currentColor',
      viewBox: '0 0 24 24'
    }, isMenuOpen ? 
      React.createElement('path', {
        strokeLinecap: 'round',
        strokeLinejoin: 'round',
        strokeWidth: 2,
        d: 'M6 18L18 6M6 6l12 12'
      }) :
      React.createElement('path', {
        strokeLinecap: 'round',
        strokeLinejoin: 'round',
        strokeWidth: 2,
        d: 'M4 6h16M4 12h16M4 18h16'
      })
    )))
  ]), 

  // Mobile Navigation Menu
  isMenuOpen ? React.createElement('div', {
    className: 'md:hidden py-4 border-t border-gray-200 dark:border-gray-700'
  }, React.createElement('div', {
    className: 'space-y-2'
  }, navItems.map((item) =>
    React.createElement(Link, {
      key: item.href,
      href: item.href,
      onClick: () => setIsMenuOpen(false),
      className: `block px-4 py-3 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center space-x-3 ${
        pathname === item.href
          ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
          : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white'
      }`
    }, [
      React.createElement('span', { key: 'icon' }, item.icon),
      React.createElement('span', { key: 'label' }, item.label)
    ])
  ))) : null));
};

export default Navigation;
