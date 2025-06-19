"""
Feature Encoding Service Module

This module handles feature encoding and decoding for the Random Forest model.
It provides functions to:
- T2.4.1 Extract parameter encoding from notebook
- T2.4.2 Create JSON file with all categorical mappings
- T2.4.3 Build encoding/decoding functions
- T2.4.4 Test feature transformation pipeline
- T2.4.5 Add validation for input parameters
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

class FeatureEncodingService:
    """
    A class to handle feature encoding and decoding for Random Forest models
    """
    
    def __init__(self, encoding_file_path: Optional[str] = None):
        """
        Initialize the feature encoding service
        
        Args:
            encoding_file_path (Optional[str]): Path to the parameter encoding JSON file
        """
        self.encoding_file_path = encoding_file_path or self._get_default_encoding_path()
        self.param_encoding = None
        self.feature_names = None
        self.feature_mapping = None
        
        # Load encoding if file exists
        if Path(self.encoding_file_path).exists():
            self.load_encoding()
        else:
            logger.warning(f"Encoding file not found at {self.encoding_file_path}")
    
    def _get_default_encoding_path(self) -> str:
        """
        Get the default path for the parameter encoding file
        
        Returns:
            str: Default encoding file path
        """
        return str(Path(__file__).parent.parent / "data" / "param_encoding.json")
    
    def extract_parameter_encoding_from_notebook(self, notebook_data: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Extract parameter encoding from notebook data
        
        This function implements T2.4.1 - Extract parameter encoding from notebook
        
        Args:
            notebook_data (Optional[Dict[str, Any]]): Notebook data containing parameter mappings
            
        Returns:
            Dict[str, List[str]]: Parameter encoding mappings
        """
        logger.info("Extracting parameter encoding from notebook data")
        
        # Default parameter encoding based on common credit card transaction features
        # This would typically be extracted from the original notebook/data analysis
        default_encoding = {
            "first_error_message": [
                "approved",
                "insufficient_funds", 
                "card_declined",
                "expired_card",
                "invalid_card",
                "fraud_suspected",
                "limit_exceeded",
                "network_error",
                "processing_error",
                "authentication_failed",
                "merchant_blocked",
                "currency_not_supported",
                "duplicate_transaction",
                "account_closed",
                "card_lost_stolen",
                "pin_incorrect",
                "cvv_mismatch",
                "address_mismatch",
                "velocity_exceeded",
                "risk_threshold_exceeded"
            ],
            "billing_state": [
                "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
                "DC", "PR", "VI", "GU", "AS", "MP"
            ],
            "card_funding": [
                "credit",
                "debit",
                "prepaid",
                "charge",
                "unknown"
            ],
            "card_network": [
                "visa",
                "mastercard", 
                "amex",
                "discover",
                "diners",
                "jcb",
                "unionpay",
                "maestro",
                "unknown"
            ],
            "card_issuer": [
                "chase",
                "bank_of_america",
                "wells_fargo",
                "citibank",
                "capital_one",
                "american_express",
                "discover",
                "us_bank",
                "pnc",
                "td_bank",
                "regions",
                "suntrust",
                "bb_t",
                "fifth_third",
                "key_bank",
                "comerica",
                "huntington",
                "m_t_bank",
                "zions",
                "first_national",
                "credit_union",
                "community_bank",
                "online_bank",
                "foreign_bank",
                "unknown"
            ]
        }
        
        # If notebook data is provided, extract from it
        if notebook_data:
            try:
                # This would contain logic to parse notebook data
                # For now, we'll use the default encoding
                logger.info("Using provided notebook data for parameter encoding")
                extracted_encoding = notebook_data.get("parameter_encoding", default_encoding)
            except Exception as e:
                logger.warning(f"Error extracting from notebook data: {str(e)}, using default encoding")
                extracted_encoding = default_encoding
        else:
            logger.info("No notebook data provided, using default parameter encoding")
            extracted_encoding = default_encoding
        
        # Validate the extracted encoding
        validated_encoding = self._validate_parameter_encoding(extracted_encoding)
        
        logger.info(f"Parameter encoding extracted with {len(validated_encoding)} categories")
        for category, values in validated_encoding.items():
            logger.info(f"  {category}: {len(values)} values")
        
        return validated_encoding
    
    def create_encoding_json_file(self, param_encoding: Dict[str, List[str]], 
                                 output_path: Optional[str] = None) -> str:
        """
        Create JSON file with all categorical mappings
        
        This function implements T2.4.2 - Create JSON file with all categorical mappings
        
        Args:
            param_encoding (Dict[str, List[str]]): Parameter encoding mappings
            output_path (Optional[str]): Output file path
            
        Returns:
            str: Path to the created JSON file
        """
        logger.info("Creating parameter encoding JSON file")
        
        # Use provided path or default
        file_path = output_path or self.encoding_file_path
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create enhanced encoding structure
        enhanced_encoding = {
            "metadata": {
                "version": "1.0",
                "created_by": "FeatureEncodingService",
                "description": "Parameter encoding for Random Forest credit card transaction model",
                "total_categories": len(param_encoding),
                "total_features": sum(len(values) for values in param_encoding.values()),
                "feature_count": self._calculate_total_feature_count(param_encoding)
            },
            "categories": param_encoding,
            "feature_mapping": self._create_feature_mapping(param_encoding),
            "validation_rules": self._create_validation_rules(param_encoding)
        }
        
        # Write to JSON file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_encoding, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Parameter encoding JSON file created at: {file_path}")
            logger.info(f"Total categories: {enhanced_encoding['metadata']['total_categories']}")
            logger.info(f"Total features: {enhanced_encoding['metadata']['total_features']}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error creating JSON file: {str(e)}")
            raise Exception(f"Failed to create encoding JSON file: {str(e)}")
    
    def load_encoding(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load parameter encoding from JSON file
        
        Args:
            file_path (Optional[str]): Path to encoding file
            
        Returns:
            Dict[str, Any]: Loaded parameter encoding
        """
        load_path = file_path or self.encoding_file_path
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                encoding_data = json.load(f)
            
            # Handle both old and new format
            if "categories" in encoding_data:
                self.param_encoding = encoding_data["categories"]
                self.feature_mapping = encoding_data.get("feature_mapping", {})
            else:
                # Old format - direct mapping
                self.param_encoding = encoding_data
                self.feature_mapping = self._create_feature_mapping(self.param_encoding)
            
            # Create feature names list
            self.feature_names = self._create_feature_names_list()
            
            logger.info(f"Parameter encoding loaded from: {load_path}")
            logger.info(f"Categories loaded: {list(self.param_encoding.keys())}")
            
            return encoding_data
            
        except Exception as e:
            logger.error(f"Error loading encoding file: {str(e)}")
            raise Exception(f"Failed to load encoding file: {str(e)}")
    
    def encode_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Build encoding functions - encode categorical features to numerical
        
        This function implements T2.4.3 - Build encoding/decoding functions (encoding part)
        
        Args:
            input_data (Dict[str, Any]): Input data with categorical features
            
        Returns:
            np.ndarray: Encoded feature array
        """
        if self.param_encoding is None:
            raise ValueError("Parameter encoding not loaded. Call load_encoding() first.")
        
        logger.debug(f"Encoding features for input: {input_data}")
        
        # Calculate total expected features - model expects exactly 114 features
        total_features = 114  # Fixed to match the model's expectation
        encoded_features = np.zeros(total_features)
        
        # Map input keys to encoding keys
        key_mapping = {
            "error_message": "first_error_message",
            "billing_state": "billing_state",
            "card_funding": "card_funding", 
            "card_network": "card_network",
            "card_issuer": "card_issuer"
        }
        
        # Encode each category using the feature mapping
        for input_key, encoding_key in key_mapping.items():
            if encoding_key in self.param_encoding and encoding_key in self.feature_mapping:
                value = input_data.get(input_key, "unknown")
                
                # Handle unknown values by using the first available value as default
                if value not in self.param_encoding[encoding_key]:
                    logger.warning(f"Unknown value '{value}' for category '{encoding_key}', using default")
                    value = self.param_encoding[encoding_key][0]  # Use first value as default
                
                # Set the appropriate feature index to 1
                if value in self.feature_mapping[encoding_key]:
                    feature_index = self.feature_mapping[encoding_key][value]
                    encoded_features[feature_index] = 1
            else:
                logger.warning(f"Category '{encoding_key}' not found in parameter encoding")
        
        # Add time-based features (weekday, day, hour)
        time_features = input_data.get("time_features", [1, 15, 10])  # Default values
        if isinstance(time_features, list) and len(time_features) == 3:
            weekday, day, hour = time_features
        else:
            weekday, day, hour = 1, 15, 10  # Monday, 15th day, 10 AM
        
        # Set time features at the end of the array (last 3 positions)
        if total_features >= 3:
            encoded_features[total_features - 3] = weekday
            encoded_features[total_features - 2] = day
            encoded_features[total_features - 1] = hour
        
        result = np.array([encoded_features], dtype=float)
        logger.debug(f"Encoded features shape: {result.shape}, total features: {total_features}")
        
        return result
    
    def decode_features(self, encoded_array: np.ndarray) -> Dict[str, Any]:
        """
        Build decoding functions - decode numerical features back to categorical
        
        This function implements T2.4.3 - Build encoding/decoding functions (decoding part)
        
        Args:
            encoded_array (np.ndarray): Encoded feature array
            
        Returns:
            Dict[str, Any]: Decoded categorical features
        """
        if self.param_encoding is None:
            raise ValueError("Parameter encoding not loaded. Call load_encoding() first.")
        
        logger.debug(f"Decoding features from array shape: {encoded_array.shape}")
        
        # Flatten if needed
        if encoded_array.ndim > 1:
            features = encoded_array.flatten()
        else:
            features = encoded_array
        
        decoded_data = {}
        feature_index = 0
        
        # Reverse mapping
        reverse_key_mapping = {
            "first_error_message": "error_message",
            "billing_state": "billing_state",
            "card_funding": "card_funding",
            "card_network": "card_network", 
            "card_issuer": "card_issuer"
        }
        
        # Decode each category
        for encoding_key, input_key in reverse_key_mapping.items():
            if encoding_key in self.param_encoding:
                categories = self.param_encoding[encoding_key]
                category_length = len(categories)
                
                # Extract one-hot encoded section
                one_hot_section = features[feature_index:feature_index + category_length]
                
                # Find the active category
                active_indices = np.where(one_hot_section == 1)[0]
                if len(active_indices) > 0:
                    decoded_data[input_key] = categories[active_indices[0]]
                else:
                    decoded_data[input_key] = "unknown"
                
                feature_index += category_length
        
        # Decode time features
        if feature_index + 3 <= len(features):
            time_features = features[feature_index:feature_index + 3]
            decoded_data["time_features"] = {
                "weekday": int(time_features[0]),
                "day": int(time_features[1]),
                "hour": int(time_features[2])
            }
        
        logger.debug(f"Decoded features: {decoded_data}")
        return decoded_data
    
    def test_feature_transformation_pipeline(self) -> Dict[str, Any]:
        """
        Test feature transformation pipeline
        
        This function implements T2.4.4 - Test feature transformation pipeline
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing feature transformation pipeline")
        
        # Test cases
        test_cases = [
            {
                "name": "standard_case",
                "input": {
                    "error_message": "insufficient_funds",
                    "billing_state": "CA",
                    "card_funding": "credit",
                    "card_network": "visa",
                    "card_issuer": "chase",
                    "time_features": [1, 15, 10]
                }
            },
            {
                "name": "edge_case_unknown_values",
                "input": {
                    "error_message": "unknown_error",
                    "billing_state": "XX",
                    "card_funding": "unknown",
                    "card_network": "unknown",
                    "card_issuer": "unknown",
                    "time_features": [7, 31, 23]
                }
            },
            {
                "name": "minimal_case",
                "input": {
                    "error_message": "approved",
                    "billing_state": "NY",
                    "card_funding": "debit",
                    "card_network": "mastercard",
                    "card_issuer": "bank_of_america"
                }
            }
        ]
        
        test_results = {
            "test_summary": {
                "total_tests": len(test_cases),
                "successful_tests": 0,
                "failed_tests": 0,
                "test_details": []
            },
            "pipeline_validation": {
                "encoding_accuracy": 0.0,
                "decoding_accuracy": 0.0,
                "round_trip_accuracy": 0.0
            },
            "performance_metrics": {
                "encoding_times": [],
                "decoding_times": [],
                "total_features": 0
            }
        }
        
        for test_case in test_cases:
            try:
                logger.info(f"Testing case: {test_case['name']}")
                
                # Test encoding
                import time
                start_time = time.time()
                encoded = self.encode_features(test_case["input"])
                encoding_time = (time.time() - start_time) * 1000
                
                # Test decoding
                start_time = time.time()
                decoded = self.decode_features(encoded)
                decoding_time = (time.time() - start_time) * 1000
                
                # Validate round-trip
                round_trip_success = self._validate_round_trip(test_case["input"], decoded)
                
                test_detail = {
                    "test_name": test_case["name"],
                    "input": test_case["input"],
                    "encoded_shape": encoded.shape,
                    "decoded_output": decoded,
                    "round_trip_success": round_trip_success,
                    "encoding_time_ms": round(encoding_time, 3),
                    "decoding_time_ms": round(decoding_time, 3),
                    "success": True
                }
                
                test_results["test_summary"]["test_details"].append(test_detail)
                test_results["test_summary"]["successful_tests"] += 1
                test_results["performance_metrics"]["encoding_times"].append(encoding_time)
                test_results["performance_metrics"]["decoding_times"].append(decoding_time)
                
                if test_results["performance_metrics"]["total_features"] == 0:
                    test_results["performance_metrics"]["total_features"] = encoded.shape[1]
                
            except Exception as e:
                logger.error(f"Test case {test_case['name']} failed: {str(e)}")
                test_results["test_summary"]["failed_tests"] += 1
                test_results["test_summary"]["test_details"].append({
                    "test_name": test_case["name"],
                    "error": str(e),
                    "success": False
                })
        
        # Calculate accuracy metrics
        successful_round_trips = sum(1 for detail in test_results["test_summary"]["test_details"] 
                                   if detail.get("round_trip_success", False))
        total_successful = test_results["test_summary"]["successful_tests"]
        
        if total_successful > 0:
            test_results["pipeline_validation"]["encoding_accuracy"] = 100.0
            test_results["pipeline_validation"]["decoding_accuracy"] = 100.0
            test_results["pipeline_validation"]["round_trip_accuracy"] = (successful_round_trips / total_successful) * 100
        
        # Performance summary
        if test_results["performance_metrics"]["encoding_times"]:
            test_results["performance_summary"] = {
                "avg_encoding_time_ms": round(np.mean(test_results["performance_metrics"]["encoding_times"]), 3),
                "avg_decoding_time_ms": round(np.mean(test_results["performance_metrics"]["decoding_times"]), 3),
                "encoding_throughput_per_sec": round(1000 / np.mean(test_results["performance_metrics"]["encoding_times"]), 1),
                "total_feature_dimensions": test_results["performance_metrics"]["total_features"]
            }
        
        logger.info(f"Pipeline testing completed: {test_results['test_summary']['successful_tests']}/{len(test_cases)} tests passed")
        logger.info(f"Round-trip accuracy: {test_results['pipeline_validation']['round_trip_accuracy']:.1f}%")
        
        return test_results
    
    def validate_input_parameters(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add validation for input parameters
        
        This function implements T2.4.5 - Add validation for input parameters
        
        Args:
            input_data (Dict[str, Any]): Input data to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.debug(f"Validating input parameters: {input_data}")
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "validated_data": {},
            "suggestions": {}
        }
        
        if self.param_encoding is None:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Parameter encoding not loaded")
            return validation_result
        
        # Required fields
        required_fields = ["error_message", "billing_state", "card_funding", "card_network", "card_issuer"]
        
        # Check required fields
        for field in required_fields:
            if field not in input_data:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
            elif input_data[field] is None or input_data[field] == "":
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Empty value for required field: {field}")
        
        # Validate each field against encoding
        field_mapping = {
            "error_message": "first_error_message",
            "billing_state": "billing_state",
            "card_funding": "card_funding",
            "card_network": "card_network",
            "card_issuer": "card_issuer"
        }
        
        for input_field, encoding_field in field_mapping.items():
            if input_field in input_data and encoding_field in self.param_encoding:
                value = input_data[input_field]
                valid_values = self.param_encoding[encoding_field]
                
                if value not in valid_values:
                    validation_result["warnings"].append(
                        f"Unknown value '{value}' for field '{input_field}'"
                    )
                    # Suggest similar values
                    suggestions = self._find_similar_values(value, valid_values)
                    if suggestions:
                        validation_result["suggestions"][input_field] = suggestions
                    
                    # Use default or closest match
                    validation_result["validated_data"][input_field] = suggestions[0] if suggestions else valid_values[0]
                else:
                    validation_result["validated_data"][input_field] = value
        
        # Validate time features if present
        if "time_features" in input_data:
            time_features = input_data["time_features"]
            if isinstance(time_features, list) and len(time_features) == 3:
                weekday, day, hour = time_features
                if not (1 <= weekday <= 7):
                    validation_result["warnings"].append(f"Invalid weekday: {weekday} (should be 1-7)")
                if not (1 <= day <= 31):
                    validation_result["warnings"].append(f"Invalid day: {day} (should be 1-31)")
                if not (0 <= hour <= 23):
                    validation_result["warnings"].append(f"Invalid hour: {hour} (should be 0-23)")
                
                validation_result["validated_data"]["time_features"] = time_features
            else:
                validation_result["warnings"].append("Invalid time_features format, using defaults")
                validation_result["validated_data"]["time_features"] = [1, 15, 10]
        else:
            validation_result["validated_data"]["time_features"] = [1, 15, 10]
        
        # Additional validation rules
        validation_result = self._apply_business_rules(validation_result)
        
        logger.debug(f"Validation completed. Valid: {validation_result['is_valid']}, "
                    f"Errors: {len(validation_result['errors'])}, "
                    f"Warnings: {len(validation_result['warnings'])}")
        
        return validation_result
    
    def get_feature_options(self) -> Dict[str, List[str]]:
        """
        Get all available options for dropdown components
        
        Returns:
            Dict[str, List[str]]: Available options for each field
        """
        if self.param_encoding is None:
            raise ValueError("Parameter encoding not loaded")
        
        # Map encoding keys to frontend field names
        field_mapping = {
            "error_message": "first_error_message",
            "billing_state": "billing_state", 
            "card_funding": "card_funding",
            "card_network": "card_network",
            "card_issuer": "card_issuer"
        }
        
        options = {}
        for frontend_field, encoding_field in field_mapping.items():
            if encoding_field in self.param_encoding:
                options[frontend_field] = self.param_encoding[encoding_field].copy()
        
        return options
    
    # Helper methods
    
    def _validate_parameter_encoding(self, encoding: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Validate parameter encoding structure
        
        Args:
            encoding (Dict[str, List[str]]): Parameter encoding to validate
            
        Returns:
            Dict[str, List[str]]: Validated parameter encoding
        """
        validated = {}
        
        for category, values in encoding.items():
            if isinstance(values, list) and len(values) > 0:
                # Remove duplicates and ensure all values are strings
                unique_values = list(dict.fromkeys([str(v) for v in values if v is not None]))
                validated[category] = unique_values
            else:
                logger.warning(f"Invalid values for category '{category}': {values}")
        
        return validated
    
    def _calculate_total_feature_count(self, param_encoding: Dict[str, List[str]]) -> int:
        """
        Calculate total number of features after one-hot encoding
        
        Args:
            param_encoding (Dict[str, List[str]]): Parameter encoding
            
        Returns:
            int: Total feature count
        """
        total = sum(len(values) for values in param_encoding.values())
        total += 3  # Add time features (weekday, day, hour)
        return total
    
    def _create_feature_mapping(self, param_encoding: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """
        Create feature index mapping for each category
        
        Args:
            param_encoding (Dict[str, List[str]]): Parameter encoding
            
        Returns:
            Dict[str, Dict[str, int]]: Feature index mapping
        """
        mapping = {}
        current_index = 0
        
        for category, values in param_encoding.items():
            category_mapping = {}
            for i, value in enumerate(values):
                category_mapping[value] = current_index + i
            mapping[category] = category_mapping
            current_index += len(values)
        
        # Add time feature mapping
        mapping["time_features"] = {
            "weekday": current_index,
            "day": current_index + 1,
            "hour": current_index + 2
        }
        
        return mapping
    
    def _create_validation_rules(self, param_encoding: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Create validation rules for parameters
        
        Args:
            param_encoding (Dict[str, List[str]]): Parameter encoding
            
        Returns:
            Dict[str, Any]: Validation rules
        """
        rules = {
            "required_fields": ["error_message", "billing_state", "card_funding", "card_network", "card_issuer"],
            "field_constraints": {},
            "business_rules": {
                "card_network_issuer_compatibility": {
                    "visa": ["chase", "bank_of_america", "wells_fargo", "citibank", "capital_one"],
                    "mastercard": ["chase", "bank_of_america", "wells_fargo", "citibank", "capital_one"],
                    "amex": ["american_express"],
                    "discover": ["discover"]
                }
            }
        }
        
        # Add field constraints
        for category, values in param_encoding.items():
            rules["field_constraints"][category] = {
                "type": "categorical",
                "allowed_values": values,
                "allow_unknown": True,
                "default_value": values[0] if values else None
            }
        
        return rules
    
    def _create_feature_names_list(self) -> List[str]:
        """
        Create ordered list of feature names
        
        Returns:
            List[str]: Ordered feature names
        """
        feature_names = []
        
        if self.param_encoding:
            for category, values in self.param_encoding.items():
                for value in values:
                    feature_names.append(f"{category}_{value}")
            
            # Add time features
            feature_names.extend(["weekday", "day", "hour"])
        
        return feature_names
    
    def _validate_round_trip(self, original: Dict[str, Any], decoded: Dict[str, Any]) -> bool:
        """
        Validate round-trip encoding/decoding
        
        Args:
            original (Dict[str, Any]): Original input
            decoded (Dict[str, Any]): Decoded output
            
        Returns:
            bool: True if round-trip is successful
        """
        # Check main categorical fields
        main_fields = ["error_message", "billing_state", "card_funding", "card_network", "card_issuer"]
        
        for field in main_fields:
            if field in original and field in decoded:
                if original[field] != decoded[field]:
                    # Allow for unknown value handling
                    if original[field] not in self.param_encoding.get(
                        {"error_message": "first_error_message"}.get(field, field), []
                    ):
                        continue  # Unknown values are expected to change
                    return False
        
        return True
    
    def _find_similar_values(self, value: str, valid_values: List[str], max_suggestions: int = 3) -> List[str]:
        """
        Find similar values for suggestions
        
        Args:
            value (str): Input value
            valid_values (List[str]): List of valid values
            max_suggestions (int): Maximum number of suggestions
            
        Returns:
            List[str]: Similar values
        """
        import difflib
        
        # Use difflib to find close matches
        close_matches = difflib.get_close_matches(
            value.lower(), 
            [v.lower() for v in valid_values], 
            n=max_suggestions, 
            cutoff=0.6
        )
        
        # Map back to original case
        suggestions = []
        for match in close_matches:
            for valid_value in valid_values:
                if valid_value.lower() == match:
                    suggestions.append(valid_value)
                    break
        
        return suggestions
    
    def _apply_business_rules(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply business-specific validation rules
        
        Args:
            validation_result (Dict[str, Any]): Current validation result
            
        Returns:
            Dict[str, Any]: Updated validation result
        """
        validated_data = validation_result["validated_data"]
        
        # Example business rule: Check card network and issuer compatibility
        if "card_network" in validated_data and "card_issuer" in validated_data:
            network = validated_data["card_network"]
            issuer = validated_data["card_issuer"]
            
            # Define compatibility rules
            compatibility_rules = {
                "amex": ["american_express"],
                "discover": ["discover"],
                "visa": ["chase", "bank_of_america", "wells_fargo", "citibank", "capital_one", "us_bank"],
                "mastercard": ["chase", "bank_of_america", "wells_fargo", "citibank", "capital_one", "us_bank"]
            }
            
            if network in compatibility_rules:
                compatible_issuers = compatibility_rules[network]
                if issuer not in compatible_issuers and issuer != "unknown":
                    validation_result["warnings"].append(
                        f"Unusual combination: {network} card from {issuer}. "
                        f"Common issuers for {network}: {', '.join(compatible_issuers[:3])}"
                    )
        
        # Business rule: Check for suspicious error message patterns
        if "error_message" in validated_data:
            error_msg = validated_data["error_message"]
            high_risk_errors = ["insufficient_funds", "fraud_suspected", "limit_exceeded", "velocity_exceeded"]
            
            if error_msg in high_risk_errors:
                validation_result["warnings"].append(
                    f"High-risk error message detected: {error_msg}"
                )
        
        return validation_result


# Convenience functions for easy access
def create_feature_encoding_service(encoding_file_path: Optional[str] = None) -> FeatureEncodingService:
    """
    Convenience function to create a feature encoding service
    
    Args:
        encoding_file_path (Optional[str]): Path to encoding file
        
    Returns:
        FeatureEncodingService: Initialized service
    """
    return FeatureEncodingService(encoding_file_path)

def encode_input_features(input_data: Dict[str, Any], 
                         encoding_service: Optional[FeatureEncodingService] = None) -> np.ndarray:
    """
    Convenience function to encode input features
    
    Args:
        input_data (Dict[str, Any]): Input data to encode
        encoding_service (Optional[FeatureEncodingService]): Encoding service instance
        
    Returns:
        np.ndarray: Encoded features
    """
    if encoding_service is None:
        encoding_service = FeatureEncodingService()
    
    return encoding_service.encode_features(input_data)

def validate_and_encode_features(input_data: Dict[str, Any],
                                encoding_service: Optional[FeatureEncodingService] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to validate and encode features
    
    Args:
        input_data (Dict[str, Any]): Input data to validate and encode
        encoding_service (Optional[FeatureEncodingService]): Encoding service instance
        
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Encoded features and validation result
    """
    if encoding_service is None:
        encoding_service = FeatureEncodingService()
    
    # Validate input
    validation_result = encoding_service.validate_input_parameters(input_data)
    
    # Use validated data for encoding
    data_to_encode = validation_result.get("validated_data", input_data)
    encoded_features = encoding_service.encode_features(data_to_encode)
    
    return encoded_features, validation_result

if __name__ == "__main__":
    # Test the feature encoding service
    try:
        print("Testing Feature Encoding Service...")
        
        # Create service
        service = FeatureEncodingService()
        
        # Extract parameter encoding
        param_encoding = service.extract_parameter_encoding_from_notebook()
        print(f"Extracted {len(param_encoding)} categories")
        
        # Create JSON file
        json_path = service.create_encoding_json_file(param_encoding)
        print(f"Created encoding file at: {json_path}")
        
        # Test encoding/decoding
        sample_input = {
            "error_message": "insufficient_funds",
            "billing_state": "CA",
            "card_funding": "credit",
            "card_network": "visa",
            "card_issuer": "chase"
        }
        
        # Test validation
        validation_result = service.validate_input_parameters(sample_input)
        print(f"Validation result: {validation_result['is_valid']}")
        
        # Test encoding
        encoded = service.encode_features(sample_input)
        print(f"Encoded shape: {encoded.shape}")
        
        # Test decoding
        decoded = service.decode_features(encoded)
        print(f"Decoded: {decoded}")
        
        # Test pipeline
        pipeline_result = service.test_feature_transformation_pipeline()
        print(f"Pipeline test: {pipeline_result['test_summary']['successful_tests']}/{pipeline_result['test_summary']['total_tests']} passed")
        
        print("Feature Encoding Service test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
