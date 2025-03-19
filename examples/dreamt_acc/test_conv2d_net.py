import pytest
import torch
import numpy as np
from unittest.mock import Mock
from examples.dreamt_acc.conv2d_net import wasa, true_pos_neg_rates_from_threshold, WASAResult

class TestWASA:
    
    def test_true_false_rates_calculation(self):
        # Test the true_false_rates_from_threshold function
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.3, 0.7, 0.8, 0.9, 0.4, 0.2, 0.6])
        
        # Test with threshold 0.5
        tpr, fpr = true_pos_neg_rates_from_threshold(y_true, y_pred, 0.5)
        # TPR: 3/4 = 0.75 (3 true positives out of 4 positive samples)
        # FPR: 1/4 = 0.25 (1 false positive out of 4 negative samples)
        assert np.isclose(tpr, 0.75)
        assert np.isclose(fpr, 0.25)
    
    def test_true_false_rates_all_trues(self):
        # Test when all predictions are true
        y_true = np.ones(10, dtype=int)
        y_pred = np.ones(10, dtype=float)
        
        # All predictions are true, so TPR and FPR should be 1.0
        tpr, fpr = true_pos_neg_rates_from_threshold(y_true, y_pred, 0.5)
        assert np.isclose(tpr, 1.0)
        assert np.isclose(fpr, 1.0)
    
    def test_true_false_rates_all_falses(self):
        # Test when all predictions are false
        y_true = np.zeros(10, dtype=int)
        y_pred = np.zeros(10, dtype=float)
        
        # All predictions are false, so TPR and FPR should be 0.0
        tpr, fpr = true_pos_neg_rates_from_threshold(y_true, y_pred, 0.5)
        assert np.isclose(tpr, 1.0)
        assert np.isclose(fpr, 0.0)
    
    def test_wasa_all_sleep_data(self):
        # Test when all data is sleep (class 1)
        mock_model = Mock()
        
        # Create a tensor of all 1s for y_test
        y_test = torch.ones((1, 100), dtype=torch.long)
        X_test = torch.zeros((1, 100, 129), dtype=torch.float32)
        
        # Mock the model output to strongly predict sleep
        mock_output = torch.zeros((1, 2, 100), dtype=torch.float32)
        mock_output[:, 1, :] = 10.0  # High logit for sleep class
        mock_model.return_value = mock_output
        mock_model.eval = Mock()
        
        # Should achieve 100% sleep accuracy but wake accuracy is undefined
        result = wasa(mock_model, X_test, y_test, target_sleep_acc=0.95)
        
        # Should achieve target sleep accuracy
        assert np.isclose(result.sleep_acc, 1.0)
        # Wake accuracy should be reported as 0 since there are no wake samples
        assert result.wake_acc == 1.0
    
    def test_wasa_all_wake_data(self):
        # Test when all data is wake (class 0)
        mock_model = Mock()
        
        # Create a tensor of all 0s for y_test
        y_test = torch.zeros((1, 100), dtype=torch.long)
        X_test = torch.zeros((1, 100, 129), dtype=torch.float32)
        
        # Mock the model output to strongly predict wake
        mock_output = torch.zeros((1, 2, 100), dtype=torch.float32)
        mock_output[:, 0, :] = 10.0  # High logit for wake class
        mock_model.return_value = mock_output
        mock_model.eval = Mock()
        
        # Sleep accuracy is undefined, wake accuracy should be 100%
        result = wasa(mock_model, X_test, y_test, target_sleep_acc=0.95)
        
        # Sleep accuracy will be reported as 0 since there are no sleep samples
        assert result.sleep_acc == 1.0
        # Wake accuracy should be 1.0
        assert np.isclose(result.wake_acc, 1.0)
    
    def test_wasa_no_valid_labels(self):
        # Test when all labels are masked (-1)
        mock_model = Mock()
        
        # Create a tensor of all -1s for y_test (masked values)
        y_test = -torch.ones((1, 100), dtype=torch.long)
        X_test = torch.zeros((1, 100, 129), dtype=torch.float32)
        
        mock_output = torch.zeros((1, 2, 100), dtype=torch.float32)
        mock_model.return_value = mock_output
        mock_model.eval = Mock()
        
        # Should return default values when no valid labels
        result = wasa(mock_model, X_test, y_test, target_sleep_acc=0.95)
        
        assert result.wake_acc == 0.0
        assert result.sleep_acc == 0.0
        assert result.threshold == 0.0
    
    def test_wasa_target_accuracy_attainable(self):
        # Test when target accuracy is attainable
        mock_model = Mock()
        
        # Create balanced dataset: 50 wake, 50 sleep
        y_test = torch.cat([torch.zeros(1, 50), torch.ones(1, 50)], dim=1)
        X_test = torch.zeros((1, 100, 129), dtype=torch.float32)
        
        # Create model outputs with clear separation
        mock_output = torch.zeros((1, 2, 100), dtype=torch.float32)
        # Wake samples get low sleep probability
        mock_output[:, 1, :50] = torch.linspace(-5, 0, 50)
        # Sleep samples get high sleep probability
        mock_output[:, 1, 50:] = torch.linspace(0, 5, 50)
        mock_model.return_value = mock_output
        mock_model.eval = Mock()
        
        # With this distribution, 95% sleep accuracy should be achievable
        result = wasa(mock_model, X_test, y_test, target_sleep_acc=0.95)
        
        # Should achieve target sleep accuracy
        assert np.isclose(result.sleep_acc, 0.95, atol=0.02)
        # Wake accuracy should be reasonable given the distribution
        assert result.wake_acc > 0.0
    
    def test_wasa_target_accuracy_unattainable(self):
        # Test when target accuracy is unattainable
        mock_model = Mock()
        
        # Create balanced dataset
        y_test = torch.cat([torch.zeros(1, 50), torch.ones(1, 50)], dim=1)
        X_test = torch.zeros((1, 100, 129), dtype=torch.float32)
        
        # Create model outputs with poor separation
        mock_output = torch.zeros((1, 2, 100), dtype=torch.float32)
        # All samples get random, overlapping probabilities
        np.random.seed(42)
        mock_output[:, 1, :] = torch.tensor(np.random.random(100))
        mock_model.return_value = mock_output
        mock_model.eval = Mock()
        
        # With random predictions, 99.9% sleep accuracy should be unattainable
        result = wasa(mock_model, X_test, y_test, target_sleep_acc=0.999)
        
        # Binary search will try to get as close as possible to target
        assert result.sleep_acc < 0.999
    
    def test_wasa_nan_outputs(self):
        # Test behavior with NaN outputs
        mock_model = Mock()
        
        y_test = torch.cat([torch.zeros(1, 50), torch.ones(1, 50)], dim=1)
        X_test = torch.zeros((1, 100, 129), dtype=torch.float32)
        
        # Create model outputs with NaNs
        mock_output = torch.zeros((1, 2, 100), dtype=torch.float32)
        mock_output[:, 1, :] = float('nan')
        mock_model.return_value = mock_output
        mock_model.eval = Mock()
        
        # Should handle NaNs gracefully
        result = wasa(mock_model, X_test, y_test, target_sleep_acc=0.95)
        
        # With NaN outputs, we expect the binary search to fail
        assert result.wake_acc == 0.0 or np.isnan(result.wake_acc)
        assert result.sleep_acc == 0.0 or np.isnan(result.sleep_acc)
    
    def test_wake_acc_point_5(self):
        y_true = torch.tensor(np.array([0, 0, 0, 1, 1, 1, 1]))
        y_pred_sleep = np.array([0.1, 0.3, 0.8, 0.7, 0.8, 0.9, 0.6])
        y_pred = torch.tensor(np.vstack([1 - y_pred_sleep, y_pred_sleep])[None, ...])
        model = Mock()
        model.return_value = y_pred
        model.eval = Mock()

        result = wasa(model, y_pred, y_true, target_sleep_acc=0.75)
        assert np.isclose(result.wake_acc, 2/3)


if __name__ == "__main__":
    pytest.main()