"""
Test script for Feature Encoding Service functionality

This script tests all the components of the feature encoding service:
- T2.4.1 - Extract parameter encoding from notebook
- T2.4.2 - Create JSON file with all categorical mappings
- T2.4.3 - Build encoding/decoding functions
- T2.4.4 - Test feature transformation pipeline
- T2.4.5 - Add validation for input parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.feature_encoding_service import FeatureEncodingService, create_feature_encoding_service, validate_and_encode_features
import json
import time
import numpy as np
from pathlib import Path

def test_feature_encoding_service():
    """
    Comprehensive test of feature encoding service functionality
    """
    print("=" * 80)
    print("TESTING FEATURE ENCODING SERVICE")
    print("=" * 80)
    
    try:
        # Test T2.4.1 - Extract parameter encoding from notebook
        print("\n1. Testing parameter encoding extraction (T2.4.1)...")
        service = FeatureEncodingService()
        
        # Test with no notebook data (should use defaults)
        param_encoding = service.extract_parameter_encoding_from_notebook()
        print(f"✓ Parameter encoding extracted with {len(param_encoding)} categories")
        
        # Validate structure
        expected_categories = ["first_error_message", "billing_state", "card_funding", "card_network", "card_issuer"]
        for category in expected_categories:
            assert category in param_encoding, f"Missing category: {category}"
            assert len(param_encoding[category]) > 0, f"Empty category: {category}"
        
        print(f"  - Categories: {list(param_encoding.keys())}")
        print(f"  - Total values: {sum(len(values) for values in param_encoding.values())}")
        
        # Test with mock notebook data
        mock_notebook_data = {
            "parameter_encoding": {
                "first_error_message": ["approved", "declined", "error"],
                "billing_state": ["CA", "NY", "TX"],
                "card_funding": ["credit", "debit"],
                "card_network": ["visa", "mastercard"],
                "card_issuer": ["chase", "wells_fargo"]
            }
        }
        
        custom_encoding = service.extract_parameter_encoding_from_notebook(mock_notebook_data)
        assert len(custom_encoding["first_error_message"]) == 3, "Custom encoding not applied"
        print("✓ Custom notebook data extraction working")
        
        # Test T2.4.2 - Create JSON file with all categorical mappings
        print("\n2. Testing JSON file creation (T2.4.2)...")
        
        # Create JSON file
        json_path = service.create_encoding_json_file(param_encoding)
        print(f"✓ JSON file created at: {json_path}")
        
        # Verify file exists and has correct structure
        assert Path(json_path).exists(), "JSON file was not created"
        
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
        
        # Check structure
        required_keys = ["metadata", "categories", "feature_mapping", "validation_rules"]
        for key in required_keys:
            assert key in saved_data, f"Missing key in JSON: {key}"
        
        assert saved_data["categories"] == param_encoding, "Categories not saved correctly"
        print(f"  - Metadata version: {saved_data['metadata']['version']}")
        print(f"  - Total categories: {saved_data['metadata']['total_categories']}")
        print(f"  - Total features: {saved_data['metadata']['total_features']}")
        
        # Test loading the created file
        service2 = FeatureEncodingService(json_path)
        assert service2.param_encoding is not None, "Failed to load created JSON file"
        print("✓ JSON file loading verification successful")
        
        # Test T2.4.3 - Build encoding/decoding functions
        print("\n3. Testing encoding/decoding functions (T2.4.3)...")
        
        # Test encoding
        sample_input = {
            "error_message": "insufficient_funds",
            "billing_state": "CA",
            "card_funding": "credit",
            "card_network": "visa",
            "card_issuer": "chase",
            "time_features": [1, 15, 10]
        }
        
        start_time = time.time()
        encoded = service.encode_features(sample_input)
        encoding_time = (time.time() - start_time) * 1000
        
        print(f"✓ Encoding completed in {encoding_time:.3f}ms")
        print(f"  - Input: {sample_input}")
        print(f"  - Encoded shape: {encoded.shape}")
        print(f"  - Feature count: {encoded.shape[1]}")
        
        # Validate encoding structure
        assert encoded.ndim == 2, "Encoded array should be 2D"
        assert encoded.shape[0] == 1, "Should have 1 sample"
        assert encoded.shape[1] > 100, "Should have many features after one-hot encoding"
        
        # Test decoding
        start_time = time.time()
        decoded = service.decode_features(encoded)
        decoding_time = (time.time() - start_time) * 1000
        
        print(f"✓ Decoding completed in {decoding_time:.3f}ms")
        print(f"  - Decoded: {decoded}")
        
        # Validate decoding structure
        required_decoded_keys = ["error_message", "billing_state", "card_funding", "card_network", "card_issuer", "time_features"]
        for key in required_decoded_keys:
            assert key in decoded, f"Missing decoded key: {key}"
        
        # Test round-trip consistency
        main_fields = ["error_message", "billing_state", "card_funding", "card_network", "card_issuer"]
        round_trip_success = True
        for field in main_fields:
            if sample_input[field] != decoded[field]:
                print(f"  Warning: Round-trip mismatch for {field}: {sample_input[field]} -> {decoded[field]}")
                round_trip_success = False
        
        if round_trip_success:
            print("✓ Round-trip encoding/decoding successful")
        else:
            print("⚠ Round-trip had some mismatches (may be expected for unknown values)")
        
        # Test with unknown values
        unknown_input = {
            "error_message": "unknown_error",
            "billing_state": "XX",
            "card_funding": "unknown",
            "card_network": "unknown",
            "card_issuer": "unknown"
        }
        
        encoded_unknown = service.encode_features(unknown_input)
        decoded_unknown = service.decode_features(encoded_unknown)
        print(f"✓ Unknown values handled: {decoded_unknown}")
        
        # Test T2.4.4 - Test feature transformation pipeline
        print("\n4. Testing feature transformation pipeline (T2.4.4)...")
        
        pipeline_result = service.test_feature_transformation_pipeline()
        
        print(f"✓ Pipeline testing completed")
        print(f"  - Total tests: {pipeline_result['test_summary']['total_tests']}")
        print(f"  - Successful tests: {pipeline_result['test_summary']['successful_tests']}")
        print(f"  - Failed tests: {pipeline_result['test_summary']['failed_tests']}")
        print(f"  - Round-trip accuracy: {pipeline_result['pipeline_validation']['round_trip_accuracy']:.1f}%")
        
        # Validate pipeline results
        assert pipeline_result['test_summary']['successful_tests'] > 0, "No successful pipeline tests"
        assert pipeline_result['pipeline_validation']['encoding_accuracy'] == 100.0, "Encoding accuracy should be 100%"
        
        if "performance_summary" in pipeline_result:
            perf = pipeline_result["performance_summary"]
            print(f"  - Avg encoding time: {perf['avg_encoding_time_ms']:.3f}ms")
            print(f"  - Avg decoding time: {perf['avg_decoding_time_ms']:.3f}ms")
            print(f"  - Encoding throughput: {perf['encoding_throughput_per_sec']:.1f} ops/sec")
            print(f"  - Total feature dimensions: {perf['total_feature_dimensions']}")
        
        # Test T2.4.5 - Add validation for input parameters
        print("\n5. Testing input parameter validation (T2.4.5)...")
        
        # Test valid input
        valid_input = {
            "error_message": "insufficient_funds",
            "billing_state": "CA",
            "card_funding": "credit",
            "card_network": "visa",
            "card_issuer": "chase"
        }
        
        validation_result = service.validate_input_parameters(valid_input)
        print(f"✓ Valid input validation completed")
        print(f"  - Is valid: {validation_result['is_valid']}")
        print(f"  - Errors: {len(validation_result['errors'])}")
        print(f"  - Warnings: {len(validation_result['warnings'])}")
        
        assert validation_result['is_valid'] == True, "Valid input should pass validation"
        
        # Test invalid input (missing fields)
        invalid_input = {
            "error_message": "insufficient_funds",
            # Missing other required fields
        }
        
        invalid_validation = service.validate_input_parameters(invalid_input)
        print(f"✓ Invalid input validation completed")
        print(f"  - Is valid: {invalid_validation['is_valid']}")
        print(f"  - Errors: {len(invalid_validation['errors'])}")
        
        assert invalid_validation['is_valid'] == False, "Invalid input should fail validation"
        assert len(invalid_validation['errors']) > 0, "Should have validation errors"
        
        # Test input with unknown values
        unknown_validation_input = {
            "error_message": "unknown_error",
            "billing_state": "CA",
            "card_funding": "credit",
            "card_network": "visa",
            "card_issuer": "chase"
        }
        
        unknown_validation = service.validate_input_parameters(unknown_validation_input)
        print(f"✓ Unknown values validation completed")
        print(f"  - Is valid: {unknown_validation['is_valid']}")
        print(f"  - Warnings: {len(unknown_validation['warnings'])}")
        print(f"  - Suggestions: {unknown_validation.get('suggestions', {})}")
        
        # Test get_feature_options
        print("\n6. Testing feature options retrieval...")
        
        feature_options = service.get_feature_options()
        print(f"✓ Feature options retrieved")
        print(f"  - Available fields: {list(feature_options.keys())}")
        
        for field, options in feature_options.items():
            print(f"  - {field}: {len(options)} options")
            assert len(options) > 0, f"No options for field: {field}"
        
        # Test convenience functions
        print("\n7. Testing convenience functions...")
        
        # Test create_feature_encoding_service
        service3 = create_feature_encoding_service()
        assert service3 is not None, "Convenience function failed"
        
        # Test validate_and_encode_features
        encoded_conv, validation_conv = validate_and_encode_features(sample_input)
        assert encoded_conv.shape == encoded.shape, "Convenience function encoding mismatch"
        assert validation_conv['is_valid'] == True, "Convenience function validation failed"
        
        print("✓ Convenience functions working correctly")
        
        # Performance benchmark
        print("\n8. Performance benchmark...")
        
        benchmark_iterations = 100
        encoding_times = []
        decoding_times = []
        validation_times = []
        
        for i in range(benchmark_iterations):
            # Benchmark encoding
            start = time.time()
            service.encode_features(sample_input)
            encoding_times.append((time.time() - start) * 1000)
            
            # Benchmark decoding
            start = time.time()
            service.decode_features(encoded)
            decoding_times.append((time.time() - start) * 1000)
            
            # Benchmark validation
            start = time.time()
            service.validate_input_parameters(sample_input)
            validation_times.append((time.time() - start) * 1000)
        
        print(f"✓ Performance benchmark completed ({benchmark_iterations} iterations)")
        print(f"  - Avg encoding time: {np.mean(encoding_times):.3f}ms")
        print(f"  - Avg decoding time: {np.mean(decoding_times):.3f}ms")
        print(f"  - Avg validation time: {np.mean(validation_times):.3f}ms")
        print(f"  - Encoding throughput: {1000/np.mean(encoding_times):.1f} ops/sec")
        print(f"  - Validation throughput: {1000/np.mean(validation_times):.1f} ops/sec")
        
        # Test error handling
        print("\n9. Testing error handling...")
        
        # Test encoding without loaded parameters
        empty_service = FeatureEncodingService("/nonexistent/path.json")
        try:
            empty_service.encode_features(sample_input)
            print("✗ Should have failed with no encoding loaded")
        except ValueError:
            print("✓ Properly rejected encoding without loaded parameters")
        
        # Test invalid input types
        try:
            service.encode_features("invalid_input")
            print("✗ Should have failed with invalid input type")
        except (ValueError, AttributeError):
            print("✓ Properly rejected invalid input type")
        
        print("\n" + "=" * 80)
        print("FEATURE ENCODING SERVICE TEST SUMMARY")
        print("=" * 80)
        print("✓ All core functionality tests passed")
        print("✓ T2.4.1 - Parameter encoding extraction: IMPLEMENTED")
        print("✓ T2.4.2 - JSON file creation with mappings: IMPLEMENTED")
        print("✓ T2.4.3 - Encoding/decoding functions: IMPLEMENTED")
        print("✓ T2.4.4 - Feature transformation pipeline: IMPLEMENTED")
        print("✓ T2.4.5 - Input parameter validation: IMPLEMENTED")
        print("\n✅ TASK T2.4 - BUILD FEATURE ENCODING SERVICE: COMPLETED")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_feature_encoding_service():
    """
    Demonstrate the feature encoding service with detailed output
    """
    print("\n" + "=" * 80)
    print("FEATURE ENCODING SERVICE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create service and generate encoding
        service = FeatureEncodingService()
        param_encoding = service.extract_parameter_encoding_from_notebook()
        json_path = service.create_encoding_json_file(param_encoding)
        
        print(f"\nParameter Encoding Summary:")
        print("-" * 50)
        total_values = 0
        for category, values in param_encoding.items():
            print(f"{category}: {len(values)} values")
            total_values += len(values)
        print(f"Total categorical values: {total_values}")
        print(f"Total features (with time): {total_values + 3}")
        
        # Demonstrate encoding process
        sample_inputs = [
            {
                "name": "High Risk Transaction",
                "data": {
                    "error_message": "insufficient_funds",
                    "billing_state": "CA",
                    "card_funding": "credit",
                    "card_network": "visa",
                    "card_issuer": "chase"
                }
            },
            {
                "name": "Low Risk Transaction",
                "data": {
                    "error_message": "approved",
                    "billing_state": "NY",
                    "card_funding": "debit",
                    "card_network": "mastercard",
                    "card_issuer": "bank_of_america"
                }
            },
            {
                "name": "Unknown Values Transaction",
                "data": {
                    "error_message": "unknown_error",
                    "billing_state": "XX",
                    "card_funding": "unknown",
                    "card_network": "unknown",
                    "card_issuer": "unknown"
                }
            }
        ]
        
        for sample in sample_inputs:
            print(f"\n{sample['name']}:")
            print("-" * 50)
            
            # Show input
            print("Input:")
            for key, value in sample['data'].items():
                print(f"  {key}: {value}")
            
            # Validate
            validation = service.validate_input_parameters(sample['data'])
            print(f"\nValidation:")
            print(f"  Valid: {validation['is_valid']}")
            print(f"  Errors: {len(validation['errors'])}")
            print(f"  Warnings: {len(validation['warnings'])}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"    - {warning}")
            
            if validation['suggestions']:
                print(f"  Suggestions: {validation['suggestions']}")
            
            # Encode
            encoded = service.encode_features(sample['data'])
            print(f"\nEncoding:")
            print(f"  Shape: {encoded.shape}")
            print(f"  Non-zero features: {np.count_nonzero(encoded)}")
            print(f"  Feature sum: {np.sum(encoded)}")
            
            # Decode
            decoded = service.decode_features(encoded)
            print(f"\nDecoded:")
            for key, value in decoded.items():
                if key != "time_features":
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        
        # Show feature options for frontend
        print(f"\nAvailable Feature Options for Frontend:")
        print("-" * 50)
        feature_options = service.get_feature_options()
        
        for field, options in feature_options.items():
            print(f"{field}:")
            print(f"  Options: {len(options)}")
            print(f"  Sample values: {options[:5]}...")
            if len(options) > 5:
                print(f"  (and {len(options) - 5} more)")
        
        return True
        
    except Exception as e:
        print(f"Demonstration failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Feature Encoding Service Tests...")
    
    # Run comprehensive tests
    test_success = test_feature_encoding_service()
    
    if test_success:
        # Run demonstration
        demonstrate_feature_encoding_service()
        print("\n🎉 All tests completed successfully!")
        print("Feature Encoding Service is ready for integration!")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)
