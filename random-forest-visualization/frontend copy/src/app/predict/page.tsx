'use client';

import * as React from 'react';
import { makePrediction, type PredictionRequest, type PredictionResponse, type IndividualPrediction } from '@/services/api';
import { useRouter } from 'next/navigation';
import PredictionResults from '@/components/PredictionResults';

// Parameter options from the encoding data
const PARAMETER_OPTIONS = {
  first_error_message: [
    { value: 'approved', label: 'Approved' },
    { value: 'insufficient_funds', label: 'Insufficient Funds' },
    { value: 'card_declined', label: 'Card Declined' },
    { value: 'expired_card', label: 'Expired Card' },
    { value: 'invalid_card', label: 'Invalid Card' },
    { value: 'fraud_suspected', label: 'Fraud Suspected' },
    { value: 'limit_exceeded', label: 'Limit Exceeded' },
    { value: 'network_error', label: 'Network Error' },
    { value: 'processing_error', label: 'Processing Error' },
    { value: 'authentication_failed', label: 'Authentication Failed' },
    { value: 'merchant_blocked', label: 'Merchant Blocked' },
    { value: 'currency_not_supported', label: 'Currency Not Supported' },
    { value: 'duplicate_transaction', label: 'Duplicate Transaction' },
    { value: 'account_closed', label: 'Account Closed' },
    { value: 'card_lost_stolen', label: 'Card Lost/Stolen' },
    { value: 'pin_incorrect', label: 'PIN Incorrect' },
    { value: 'cvv_mismatch', label: 'CVV Mismatch' },
    { value: 'address_mismatch', label: 'Address Mismatch' },
    { value: 'velocity_exceeded', label: 'Velocity Exceeded' },
    { value: 'risk_threshold_exceeded', label: 'Risk Threshold Exceeded' }
  ],
  billing_state: [
    { value: 'AL', label: 'Alabama' },
    { value: 'AK', label: 'Alaska' },
    { value: 'AZ', label: 'Arizona' },
    { value: 'AR', label: 'Arkansas' },
    { value: 'CA', label: 'California' },
    { value: 'CO', label: 'Colorado' },
    { value: 'CT', label: 'Connecticut' },
    { value: 'DE', label: 'Delaware' },
    { value: 'FL', label: 'Florida' },
    { value: 'GA', label: 'Georgia' },
    { value: 'HI', label: 'Hawaii' },
    { value: 'ID', label: 'Idaho' },
    { value: 'IL', label: 'Illinois' },
    { value: 'IN', label: 'Indiana' },
    { value: 'IA', label: 'Iowa' },
    { value: 'KS', label: 'Kansas' },
    { value: 'KY', label: 'Kentucky' },
    { value: 'LA', label: 'Louisiana' },
    { value: 'ME', label: 'Maine' },
    { value: 'MD', label: 'Maryland' },
    { value: 'MA', label: 'Massachusetts' },
    { value: 'MI', label: 'Michigan' },
    { value: 'MN', label: 'Minnesota' },
    { value: 'MS', label: 'Mississippi' },
    { value: 'MO', label: 'Missouri' },
    { value: 'MT', label: 'Montana' },
    { value: 'NE', label: 'Nebraska' },
    { value: 'NV', label: 'Nevada' },
    { value: 'NH', label: 'New Hampshire' },
    { value: 'NJ', label: 'New Jersey' },
    { value: 'NM', label: 'New Mexico' },
    { value: 'NY', label: 'New York' },
    { value: 'NC', label: 'North Carolina' },
    { value: 'ND', label: 'North Dakota' },
    { value: 'OH', label: 'Ohio' },
    { value: 'OK', label: 'Oklahoma' },
    { value: 'OR', label: 'Oregon' },
    { value: 'PA', label: 'Pennsylvania' },
    { value: 'RI', label: 'Rhode Island' },
    { value: 'SC', label: 'South Carolina' },
    { value: 'SD', label: 'South Dakota' },
    { value: 'TN', label: 'Tennessee' },
    { value: 'TX', label: 'Texas' },
    { value: 'UT', label: 'Utah' },
    { value: 'VT', label: 'Vermont' },
    { value: 'VA', label: 'Virginia' },
    { value: 'WA', label: 'Washington' },
    { value: 'WV', label: 'West Virginia' },
    { value: 'WI', label: 'Wisconsin' },
    { value: 'WY', label: 'Wyoming' },
    { value: 'DC', label: 'District of Columbia' },
    { value: 'PR', label: 'Puerto Rico' }
  ],
  card_funding: [
    { value: 'credit', label: 'Credit' },
    { value: 'debit', label: 'Debit' },
    { value: 'prepaid', label: 'Prepaid' },
    { value: 'charge', label: 'Charge' },
    { value: 'unknown', label: 'Unknown' }
  ],
  card_network: [
    { value: 'visa', label: 'Visa' },
    { value: 'mastercard', label: 'Mastercard' },
    { value: 'amex', label: 'American Express' },
    { value: 'discover', label: 'Discover' },
    { value: 'diners', label: 'Diners Club' },
    { value: 'jcb', label: 'JCB' },
    { value: 'unionpay', label: 'UnionPay' },
    { value: 'maestro', label: 'Maestro' },
    { value: 'unknown', label: 'Unknown' }
  ],
  card_issuer: [
    { value: 'chase', label: 'Chase' },
    { value: 'bank_of_america', label: 'Bank of America' },
    { value: 'wells_fargo', label: 'Wells Fargo' },
    { value: 'citibank', label: 'Citibank' },
    { value: 'capital_one', label: 'Capital One' },
    { value: 'american_express', label: 'American Express' },
    { value: 'discover', label: 'Discover' },
    { value: 'us_bank', label: 'US Bank' },
    { value: 'pnc', label: 'PNC Bank' },
    { value: 'td_bank', label: 'TD Bank' },
    { value: 'regions', label: 'Regions Bank' },
    { value: 'suntrust', label: 'SunTrust' },
    { value: 'bb_t', label: 'BB&T' },
    { value: 'fifth_third', label: 'Fifth Third Bank' },
    { value: 'key_bank', label: 'KeyBank' },
    { value: 'comerica', label: 'Comerica' },
    { value: 'huntington', label: 'Huntington Bank' },
    { value: 'm_t_bank', label: 'M&T Bank' },
    { value: 'zions', label: 'Zions Bank' },
    { value: 'first_national', label: 'First National Bank' },
    { value: 'credit_union', label: 'Credit Union' },
    { value: 'community_bank', label: 'Community Bank' },
    { value: 'online_bank', label: 'Online Bank' },
    { value: 'foreign_bank', label: 'Foreign Bank' },
    { value: 'unknown', label: 'Unknown' }
  ]
};

interface FormData {
  error_message: string;
  billing_state: string;
  card_funding: string;
  card_network: string;
  card_issuer: string;
  weekday: number;
  day: number;
  hour: number;
}

interface FormErrors {
  error_message?: string;
  billing_state?: string;
  card_funding?: string;
  card_network?: string;
  card_issuer?: string;
  weekday?: string;
  day?: string;
  hour?: string;
}

export default function PredictPage() {
  const [formData, setFormData] = React.useState<FormData>({
    error_message: '',
    billing_state: '',
    card_funding: '',
    card_network: '',
    card_issuer: '',
    weekday: 1,
    day: 15,
    hour: 10
  });
  
  const [errors, setErrors] = React.useState<FormErrors>({});
  const [isLoading, setIsLoading] = React.useState(false);
  const [predictions, setPredictions] = React.useState<PredictionResponse | null>(null);
  const [selectedTreeId, setSelectedTreeId] = React.useState<number | null>(null);
  const [router, setRouter] = React.useState<any>(null);

  React.useEffect(() => {
    // Initialize router on client side
    import('next/navigation').then(({ useRouter }) => {
      setRouter({ push: (url: string) => window.location.href = url });
    });
  }, []);

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};
    
    if (!formData.error_message) {
      newErrors.error_message = 'Error message is required';
    }
    if (!formData.billing_state) {
      newErrors.billing_state = 'Billing state is required';
    }
    if (!formData.card_funding) {
      newErrors.card_funding = 'Card funding is required';
    }
    if (!formData.card_network) {
      newErrors.card_network = 'Card network is required';
    }
    if (!formData.card_issuer) {
      newErrors.card_issuer = 'Card issuer is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setIsLoading(true);
    try {
      // Prepare the request with time_features
      const requestData = {
        error_message: formData.error_message,
        billing_state: formData.billing_state,
        card_funding: formData.card_funding,
        card_network: formData.card_network,
        card_issuer: formData.card_issuer,
        time_features: [formData.weekday, formData.day, formData.hour] as [number, number, number]
      };
      
      const response = await makePrediction(requestData);
      setPredictions(response);
    } catch (error) {
      console.error('Prediction failed:', error);
      // TODO: Add error toast notification
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  };

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
      router.push(`/decision-path?tree=${treeId}`);
    } catch (error) {
      console.error('Failed to navigate to decision path:', error);
    }
  };

  const getSuccessColor = (prediction: number): string => {
    if (prediction >= 0.7) return 'text-green-600 bg-green-50 border-green-200';
    if (prediction >= 0.5) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getSuccessLabel = (prediction: number): string => {
    if (prediction >= 0.7) return 'High Success';
    if (prediction >= 0.5) return 'Medium Success';
    return 'Low Success';
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Make Predictions
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Input transaction parameters to get predictions from all 100 trees in the Random Forest model.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Prediction Form */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
            Transaction Parameters
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Error Message */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Error Message *
              </label>
              <select 
                value={formData.error_message}
                onChange={(e) => handleInputChange('error_message', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white ${
                  errors.error_message ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              >
                <option value="">Select error message...</option>
                {PARAMETER_OPTIONS.first_error_message.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {errors.error_message && (
                <p className="mt-1 text-sm text-red-600">{errors.error_message}</p>
              )}
            </div>
            
            {/* Billing State */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Billing State *
              </label>
              <select 
                value={formData.billing_state}
                onChange={(e) => handleInputChange('billing_state', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white ${
                  errors.billing_state ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              >
                <option value="">Select billing state...</option>
                {PARAMETER_OPTIONS.billing_state.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {errors.billing_state && (
                <p className="mt-1 text-sm text-red-600">{errors.billing_state}</p>
              )}
            </div>
            
            {/* Card Funding */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Card Funding *
              </label>
              <select 
                value={formData.card_funding}
                onChange={(e) => handleInputChange('card_funding', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white ${
                  errors.card_funding ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              >
                <option value="">Select card funding...</option>
                {PARAMETER_OPTIONS.card_funding.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {errors.card_funding && (
                <p className="mt-1 text-sm text-red-600">{errors.card_funding}</p>
              )}
            </div>
            
            {/* Card Network */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Card Network *
              </label>
              <select 
                value={formData.card_network}
                onChange={(e) => handleInputChange('card_network', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white ${
                  errors.card_network ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              >
                <option value="">Select card network...</option>
                {PARAMETER_OPTIONS.card_network.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {errors.card_network && (
                <p className="mt-1 text-sm text-red-600">{errors.card_network}</p>
              )}
            </div>
            
            {/* Card Issuer */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Card Issuer *
              </label>
              <select 
                value={formData.card_issuer}
                onChange={(e) => handleInputChange('card_issuer', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white ${
                  errors.card_issuer ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              >
                <option value="">Select card issuer...</option>
                {PARAMETER_OPTIONS.card_issuer.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {errors.card_issuer && (
                <p className="mt-1 text-sm text-red-600">{errors.card_issuer}</p>
              )}
            </div>
            
            {/* Time Parameters Section */}
            <div className="border-t border-gray-200 dark:border-gray-600 pt-6">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Time Parameters
              </h3>
              
              <div className="grid grid-cols-3 gap-4">
                {/* Weekday */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Weekday
                  </label>
                  <select 
                    value={formData.weekday}
                    onChange={(e) => handleInputChange('weekday', e.target.value)}
                    className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white border-gray-300 dark:border-gray-600"
                  >
                    <option value={1}>Monday</option>
                    <option value={2}>Tuesday</option>
                    <option value={3}>Wednesday</option>
                    <option value={4}>Thursday</option>
                    <option value={5}>Friday</option>
                    <option value={6}>Saturday</option>
                    <option value={7}>Sunday</option>
                  </select>
                </div>
                
                {/* Day */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Day of Month
                  </label>
                  <input 
                    type="number"
                    min="1"
                    max="31"
                    value={formData.day}
                    onChange={(e) => handleInputChange('day', e.target.value)}
                    className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white border-gray-300 dark:border-gray-600"
                  />
                </div>
                
                {/* Hour */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Hour (24h)
                  </label>
                  <input 
                    type="number"
                    min="0"
                    max="23"
                    value={formData.hour}
                    onChange={(e) => handleInputChange('hour', e.target.value)}
                    className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white border-gray-300 dark:border-gray-600"
                  />
                </div>
              </div>
            </div>
            
            <button 
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-3 px-4 rounded-md transition-colors flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Getting Predictions...
                </>
              ) : (
                'Get Predictions from All Trees'
              )}
            </button>
          </form>
        </div>

        {/* Enhanced Results Panel using PredictionResults component */}
        <PredictionResults 
          predictions={predictions} 
          formData={formData}
          onTreeClick={handleTreeClick}
        />
      </div>

      {/* Individual Tree Predictions Section */}
      {predictions && (
        <div className="mt-12 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Individual Tree Predictions
            </h2>
            <p className="text-gray-600 dark:text-gray-300">
              Each tree in the Random Forest provides its own prediction. Click on any tree to explore its decision path.
            </p>
          </div>

          {/* Tree Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {predictions.individual_predictions.map((treePrediction: IndividualPrediction) => {
              const percentage = (treePrediction.prediction.value * 100).toFixed(1);
              const isHighSuccess = treePrediction.prediction.value >= 0.7;
              const isMediumSuccess = treePrediction.prediction.value >= 0.5;
              
              return (
                <div
                  key={treePrediction.tree_id}
                  onClick={() => handleTreeClick(treePrediction.tree_id)}
                  className={`
                    relative p-4 rounded-lg border-2 cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-lg
                    ${isHighSuccess 
                      ? 'bg-green-50 border-green-200 hover:bg-green-100 dark:bg-green-900/20 dark:border-green-700 dark:hover:bg-green-900/30' 
                      : isMediumSuccess 
                      ? 'bg-yellow-50 border-yellow-200 hover:bg-yellow-100 dark:bg-yellow-900/20 dark:border-yellow-700 dark:hover:bg-yellow-900/30'
                      : 'bg-red-50 border-red-200 hover:bg-red-100 dark:bg-red-900/20 dark:border-red-700 dark:hover:bg-red-900/30'
                    }
                    ${selectedTreeId === treePrediction.tree_id ? 'ring-2 ring-blue-500 scale-105' : ''}
                  `}
                >
                  {/* Tree Number Badge */}
                  <div className="flex items-center justify-between mb-3">
                    <div className={`
                      px-2 py-1 rounded-full text-xs font-bold
                      ${isHighSuccess 
                        ? 'bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200' 
                        : isMediumSuccess 
                        ? 'bg-yellow-200 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-200'
                        : 'bg-red-200 text-red-800 dark:bg-red-800 dark:text-red-200'
                      }
                    `}>
                      Tree {treePrediction.tree_id}
                    </div>
                    
                    {/* Success Icon */}
                    <div className={`
                      w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                      ${isHighSuccess 
                        ? 'bg-green-500 text-white' 
                        : isMediumSuccess 
                        ? 'bg-yellow-500 text-white'
                        : 'bg-red-500 text-white'
                      }
                    `}>
                      {isHighSuccess ? '✓' : isMediumSuccess ? '⚠' : '✗'}
                    </div>
                  </div>

                  {/* Probability Percentage */}
                  <div className="text-center mb-3">
                    <div className={`
                      text-2xl font-black mb-1
                      ${isHighSuccess 
                        ? 'text-green-700 dark:text-green-300' 
                        : isMediumSuccess 
                        ? 'text-yellow-700 dark:text-yellow-300'
                        : 'text-red-700 dark:text-red-300'
                      }
                    `}>
                      {percentage}%
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 font-medium">
                      Success Rate
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-3">
                    <div 
                      className={`
                        h-2 rounded-full transition-all duration-1000 ease-out
                        ${isHighSuccess 
                          ? 'bg-green-500' 
                          : isMediumSuccess 
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                        }
                      `}
                      style={{ width: `${treePrediction.prediction.value * 100}%` }}
                    />
                  </div>

                  {/* Leaf Node Info */}
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Leaf Node Probability
                    </div>
                    <div className={`
                      text-sm font-semibold
                      ${isHighSuccess 
                        ? 'text-green-600 dark:text-green-400' 
                        : isMediumSuccess 
                        ? 'text-yellow-600 dark:text-yellow-400'
                        : 'text-red-600 dark:text-red-400'
                      }
                    `}>
                      {percentage}%
                    </div>
                  </div>

                  {/* Hover Effect */}
                  <div className="absolute inset-0 bg-white/10 dark:bg-black/10 opacity-0 hover:opacity-100 transition-opacity duration-300 rounded-lg pointer-events-none" />
                  
                  {/* Click to Explore Indicator */}
                  <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                      Click to explore
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Summary Stats */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-700">
              <div className="text-green-800 dark:text-green-300 text-sm font-medium">High Success Trees</div>
              <div className="text-green-900 dark:text-green-200 text-2xl font-bold">
                {predictions.individual_predictions.filter(p => p.prediction.value >= 0.7).length}
              </div>
              <div className="text-green-600 dark:text-green-400 text-xs">≥70% success rate</div>
            </div>
            
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-700">
              <div className="text-yellow-800 dark:text-yellow-300 text-sm font-medium">Medium Success Trees</div>
              <div className="text-yellow-900 dark:text-yellow-200 text-2xl font-bold">
                {predictions.individual_predictions.filter(p => p.prediction.value >= 0.5 && p.prediction.value < 0.7).length}
              </div>
              <div className="text-yellow-600 dark:text-yellow-400 text-xs">50-69% success rate</div>
            </div>
            
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-700">
              <div className="text-red-800 dark:text-red-300 text-sm font-medium">Low Success Trees</div>
              <div className="text-red-900 dark:text-red-200 text-2xl font-bold">
                {predictions.individual_predictions.filter(p => p.prediction.value < 0.5).length}
              </div>
              <div className="text-red-600 dark:text-red-400 text-xs">&lt;50% success rate</div>
            </div>
          </div>

          {/* Instructions */}
          <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-700">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg className="w-5 h-5 text-blue-500 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300">How to Use</h4>
                <p className="mt-1 text-sm text-blue-700 dark:text-blue-400">
                  Each card represents one decision tree in the Random Forest. The percentage shows the leaf node probability 
                  for your input parameters. Click on any tree to explore its decision path and understand how it arrived at its prediction.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
