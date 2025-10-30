import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random

class Predictor:
    def __init__(self, model, model_path, device):
        self.device = device
        self.model = model.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
    def predict_single(self, image_tensor):
        """Predict a single image"""
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def predict_batch(self, data_loader):
        """Predict a batch of images and calculate accuracy"""
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        return accuracy, all_predictions, all_labels
    
    def visualize_predictions(self, data_loader, num_images=10):
        """Visualize predictions on random images"""
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        # Get random samples
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # Select random indices
        indices = random.sample(range(len(images)), num_images)
        
        for i, idx in enumerate(indices):
            image = images[idx]
            true_label = labels[idx].item()
            
            # Make prediction
            predicted_label, confidence, probs = self.predict_single(image)
            
            # Plot image
            axes[i].imshow(image.squeeze().numpy(), cmap='gray')
            axes[i].set_title(f'True: {true_label}\nPred: {predicted_label} ({confidence:.2%})', 
                            color='green' if predicted_label == true_label else 'red')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/predictions_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_classification_report(self, true_labels, predictions):
        """Print detailed classification report"""
        report = classification_report(true_labels, predictions, 
                                     target_names=[str(i) for i in range(10)])
        print("Classification Report:")
        print(report)
        return report

def load_test_data():
    """Load test dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return test_loader

if __name__ == "__main__":
    import torch
    from model import DigitRecognitionNN, AdvancedDigitCNN
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose model (make sure it matches the trained model)
    # model = DigitRecognitionNN()
    model = AdvancedDigitCNN()
    
    # Initialize predictor
    predictor = Predictor(model, 'models/best_model.pth', device)
    
    # Load test data
    test_loader = load_test_data()
    
    # Test model accuracy
    print("Testing model on test dataset...")
    accuracy, predictions, true_labels = predictor.predict_batch(test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Visualize some predictions
    print("Visualizing predictions...")
    predictor.visualize_predictions(test_loader, num_images=10)
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    predictor.plot_confusion_matrix(true_labels, predictions)
    
    # Print classification report
    predictor.print_classification_report(true_labels, predictions)