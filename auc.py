import torch
import numpy as np
from torchmetrics.classification import AUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def test_multiclass_auroc():
    """Test multiclass AUROC with various scenarios"""
    
    print("Testing Multiclass AUROC...")
    print("=" * 50)
    
    # Initialize AUROC metric (same as in your model)
    auroc_metric = AUROC(task="multiclass", num_classes=4)
    
    # Test Case 1: Perfect predictions
    print("\n1. Testing Perfect Predictions:")
    perfect_probs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # Class 0
        [0.0, 1.0, 0.0, 0.0],  # Class 1
        [0.0, 0.0, 1.0, 0.0],  # Class 2
        [0.0, 0.0, 0.0, 1.0],  # Class 3
    ])
    perfect_targets = torch.tensor([0, 1, 2, 3])
    
    perfect_auc = auroc_metric(perfect_probs, perfect_targets)
    print(f"Perfect predictions AUROC: {perfect_auc:.4f} (should be 1.0)")
    
    # Reset metric
    auroc_metric.reset()
    
    # Test Case 2: Random predictions
    print("\n2. Testing Random Predictions:")
    torch.manual_seed(42)
    random_probs = torch.softmax(torch.randn(100, 4), dim=1)  # Random softmax probs
    random_targets = torch.randint(0, 4, (100,))  # Random targets
    
    random_auc = auroc_metric(random_probs, random_targets)
    print(f"Random predictions AUROC: {random_auc:.4f} (should be ~0.5)")
    
    # Reset metric
    auroc_metric.reset()
    
    # Test Case 3: Realistic scenario (like your training)
    print("\n3. Testing Realistic Scenario:")
    torch.manual_seed(123)
    batch_size = 32
    realistic_logits = torch.randn(batch_size, 4) * 2  # Some variation
    realistic_probs = torch.softmax(realistic_logits, dim=1)
    realistic_targets = torch.randint(0, 4, (batch_size,))
    
    realistic_auc = auroc_metric(realistic_probs, realistic_targets)
    print(f"Realistic scenario AUROC: {realistic_auc:.4f}")
    
    # Test Case 4: Multiple batches (like your training loop)
    print("\n4. Testing Multiple Batches (Training Loop Simulation):")
    auroc_metric.reset()
    
    all_probs = []
    all_targets = []
    
    for batch_idx in range(5):  # Simulate 5 batches
        torch.manual_seed(batch_idx + 100)
        batch_logits = torch.randn(16, 4) + torch.tensor([0.5, -0.2, 0.1, -0.4])  # Slight bias
        batch_probs = torch.softmax(batch_logits, dim=1)
        batch_targets = torch.randint(0, 4, (16,))
        
        all_probs.append(batch_probs)
        all_targets.append(batch_targets)
        
        print(f"  Batch {batch_idx + 1}: probs shape {batch_probs.shape}, targets shape {batch_targets.shape}")
    
    # Concatenate all batches (like in your on_train_epoch_end)
    all_probs_cat = torch.cat(all_probs, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)
    
    multi_batch_auc = auroc_metric(all_probs_cat, all_targets_cat)
    print(f"Multi-batch AUROC: {multi_batch_auc:.4f}")
    
    # Test Case 5: Compare with sklearn
    print("\n5. Comparing with sklearn:")
    try:
        # Convert to numpy for sklearn
        probs_np = all_probs_cat.detach().numpy()
        targets_np = all_targets_cat.detach().numpy()
        
        sklearn_auc = roc_auc_score(targets_np, probs_np, multi_class='ovr', average='macro')
        print(f"Sklearn AUROC (macro): {sklearn_auc:.4f}")
        print(f"TorchMetrics AUROC: {multi_batch_auc:.4f}")
        print(f"Difference: {abs(sklearn_auc - multi_batch_auc.item()):.6f}")
        
    except Exception as e:
        print(f"Sklearn comparison failed: {e}")
    
    # Test Case 6: Edge cases
    print("\n6. Testing Edge Cases:")
    
    # All same class
    same_class_probs = torch.softmax(torch.randn(20, 4), dim=1)
    same_class_targets = torch.zeros(20, dtype=torch.long)  # All class 0
    
    try:
        auroc_metric.reset()
        same_class_auc = auroc_metric(same_class_probs, same_class_targets)
        print(f"All same class AUROC: {same_class_auc:.4f}")
    except Exception as e:
        print(f"All same class failed: {e}")
    
    # Very confident wrong predictions
    auroc_metric.reset()
    wrong_probs = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],  # Predicts class 3, actually class 0
        [0.0, 0.0, 1.0, 0.0],  # Predicts class 2, actually class 1  
        [0.0, 1.0, 0.0, 0.0],  # Predicts class 1, actually class 2
        [1.0, 0.0, 0.0, 0.0],  # Predicts class 0, actually class 3
    ])
    wrong_targets = torch.tensor([0, 1, 2, 3])
    
    wrong_auc = auroc_metric(wrong_probs, wrong_targets)
    print(f"Confident wrong predictions AUROC: {wrong_auc:.4f} (should be 0.0)")
    
    print("\n" + "=" * 50)
    print("AUROC Testing Complete!")

def test_your_training_pattern():
    """Test the exact pattern used in your training code"""
    print("\n\nTesting Your Exact Training Pattern:")
    print("=" * 50)
    
    # Simulate your training_step_outputs
    training_step_outputs = []
    train_auc = AUROC(task="multiclass", num_classes=4)
    
    # Simulate 10 training steps
    for step in range(10):
        torch.manual_seed(step + 200)
        batch_size = 8
        
        # Simulate model output (logits)
        y_hat = torch.randn(batch_size, 4) * 1.5
        targets = torch.randint(0, 4, (batch_size,))
        
        # Convert to probs (like in your training_step)
        probs = torch.softmax(y_hat, dim=1).detach()
        
        # Append to outputs (like in your training_step)
        training_step_outputs.append({
            'probs': probs,
            'targets': targets
        })
        
        print(f"Step {step + 1}: probs {probs.shape}, targets {targets.shape}")
    
    # Simulate your on_train_epoch_end
    all_probs = torch.cat([x['probs'] for x in training_step_outputs])
    all_targets = torch.cat([x['targets'] for x in training_step_outputs])
    
    print(f"\nConcatenated: probs {all_probs.shape}, targets {all_targets.shape}")
    
    # Calculate AUC (like in your on_train_epoch_end)
    auc = train_auc(all_probs, all_targets)
    print(f"Final epoch AUC: {auc:.4f}")
    
    # Check for any issues
    print(f"Probs range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    print(f"Probs sum per sample: {all_probs.sum(dim=1)[:5]}...")  # Should be ~1.0
    print(f"Targets range: [{all_targets.min()}, {all_targets.max()}]")
    print(f"Target distribution: {torch.bincount(all_targets)}")

if __name__ == "__main__":
    test_multiclass_auroc()
    test_your_training_pattern()