import logging
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from scripts import utils
import warnings

warnings.filterwarnings("ignore")

class ModelTrainer():
    def __init__(self, random_state=4):
        self.random_state = random_state
        logging.basicConfig(level=logging.INFO)
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
            "XGBClassifier": XGBClassifier(random_state=self.random_state),
            "CatBoosting Classifier": CatBoostClassifier(verbose=False, random_state=self.random_state),
            "AdaBoost Classifier": AdaBoostClassifier(random_state=self.random_state)
        }
        
        self.model_params = {
            "Decision Tree": {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "Random Forest": {
                'n_estimators': [32,64,128],
                #'max_depth': [None, 10, 20, 30],
                #'min_samples_split': [2, 5, 10],
                #'min_samples_leaf': [1, 2, 4]
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05],
                'n_estimators': [32,64,128],
                'subsample': [0.8, 0.9, 1]
            },
            "XGBClassifier": {
                'n_estimators': [32,64,128],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            },
            "CatBoosting Classifier": {
                'iterations': [30, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 4, 5]
            },
            "AdaBoost Classifier": {
                'n_estimators': [32,64,128],
                'learning_rate': [0.01, 0.1, 1]
            }
        }
    def initiate_model_trainer(self, 
                               train_dataset,
                               val_dataset,
                               test_dataset, 
                               root):
        logging.info("Splitting datasets into features and labels.")
        train_X, train_Y = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
        val_X, val_Y = val_dataset.iloc[:, :-1], val_dataset.iloc[:, -1]
        test_X, test_Y = test_dataset.iloc[:, :-1], test_dataset.iloc[:, -1]

        logging.info("Evaluating models.")
        best_model_name = self.evaluate_models(root, train_X, train_Y, val_X, val_Y, test_X, test_Y)                 

        logging.info(f"Loading the saved best model: {best_model_name}.")
        saved_best_model = utils.load_object(root = root, obj_name = 'best_model.pkl')

        logging.info("Evaluating the best model on the test dataset.")
        self.evaluate_and_save_results(saved_best_model, test_X, test_Y, root, best_model_name)

    def evaluate_models(self, root, train_X, train_Y, val_X, val_Y, test_X, test_Y):    
        model_scores = {}  
        best_test_accuracy = 0.0
        best_model_name = None
        best_model = None
        for model_name, model in self.models.items():
            logging.info(f"Training {model_name}...")       
            grid_search = GridSearchCV(estimator=model, param_grid=self.model_params[model_name], cv=5)
            grid_search.fit(train_X, train_Y)
            # Get the best estimator
            best_estimator = grid_search.best_estimator_
            # train the model
            best_estimator.fit(train_X, train_Y)

            # Model validation
            val_Y_pred = best_estimator.predict(val_X)
            val_accuracy = (accuracy_score(val_Y, val_Y_pred))*100
            logging.info(f'{model_name} - Validation Accuracy: {val_accuracy}')

            
            test_pred_Y = best_estimator.predict(test_X)
            """ Evaluating the model """  
            test_accuracy = (accuracy_score(test_Y, test_pred_Y))*100
            precision = precision_score(test_Y, test_pred_Y)
            recall = recall_score(test_Y, test_pred_Y)
            f1 = f1_score(test_Y, test_pred_Y)
            cm = confusion_matrix(test_Y, test_pred_Y)
            classification_rep = classification_report(test_Y, test_pred_Y)
            
            logging.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")
            logging.info(f"{model_name} - Precision: {precision:.4f}")
            logging.info(f"{model_name} - Recall: {recall:.4f}")
            logging.info(f"{model_name} - F1-score: {f1:.4f}")
            logging.info(f"{model_name} - Confusion Matrix:\n{cm}")
            logging.info(f"{model_name} - Classification Report:\n{classification_rep}")

            # Save the model with the best test accuracy
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model_name = model_name
                best_model = best_estimator

            model_scores[model_name] = test_accuracy  
        logging.info(f"Best Model: {best_model_name}")
        logging.info(f"Best Test Accuracy: {best_test_accuracy:.4f}")
        logging.info('Test Accuracy results:', model_scores)

        #save the best model
        utils.save_object(root=root, obj=best_model, obj_name='best_model.pkl')  
        return best_model_name 

    def evaluate_and_save_results(self, model, test_X, test_Y, root, model_name):
        """
        Evaluate the model on the test set and save the results.
        
        Parameters:
            model (sklearn estimator): The trained model.
            test_X (DataFrame): Test features.
            test_Y (Series): Test labels.
            root (str): Path to save the results.
            model_name (str): The name of the model.
        """
        predicted = model.predict(test_X)
        accuracy = accuracy_score(test_Y, predicted) * 100
        precision = precision_score(test_Y, predicted)
        recall = recall_score(test_Y, predicted)
        f1 = f1_score(test_Y, predicted)
        cm = confusion_matrix(test_Y, predicted)
        classification_rep = classification_report(test_Y, predicted)

        utils.save_results(
            root=root,
            model_name=model_name,
            test_accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cm=cm,
            classification_rep=classification_rep
        )
        logging.info(f"Results saved for {model_name}:")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1-score: {f1}")
        logging.info(f"Confusion Matrix:\n{cm}")
        logging.info(f"Classification Report:\n{classification_rep}")

        return accuracy