import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def save_object(root, obj, obj_name):
    """Save Python object `obj` to a pickle file in the specified directory."""
    obj_path = os.path.join(root, 'data', 'artifacts', obj_name)
    with open(obj_path, 'wb') as file:
        return pickle.dump(obj, file)
    
def load_object(root, obj_name):
    """Load a Python object from a pickle file in the specified directory."""
    file_path = os.path.join( root, 'data', 'artifacts', obj_name)
    with open(file_path, 'rb') as file:
        return pickle.load(file)  


def save_results(root, 
                 model_name,
                 test_accuracy,
                 precision,
                 recall,
                 f1,
                 cm,
                 classification_rep
                 ):
    """Save evaluation results to a text file in the specified directory."""
    file_path = os.path.join(root, 'data', 'artifacts', 'best_results.txt')
    with open(file_path, 'w') as file:
        file.write(f'The Best Model: {model_name}\n')
        file.write(f'Test Accuracy: {test_accuracy:.2f}%\n')
        file.write(f'Precision: {precision:.4f}\n')
        file.write(f'Recall: {recall:.4f}\n')
        file.write(f'F1 Score: {f1:.4f}\n\n')
        file.write('Confusion Matrix:\n')
        file.write(f'{cm}\n\n')
        file.write('Classification Report:\n')
        file.write(classification_rep)

  