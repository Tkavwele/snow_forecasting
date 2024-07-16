from pathlib import Path
import time
from scripts import transform_data, model_trainer
start_time = time.time()

# Determine root directory
root = Path(__file__).parents[0]
print(f"Root directory: {root}")

# Data Transformation Stage
print(f">>>>>> Stage 'Data Transformation' started <<<<<<")
data_transform = transform_data.DataTransformation(root=root)
train_dataset, val_dataset, test_dataset = data_transform.initiate_data_transformation()
print(f">>>>>> Stage 'Data Transformation' completed <<<<<<\n\nx==========x")


# Model Training Stage
print(f">>>>>> Stage 'Model Training' started <<<<<<")
trainer = model_trainer.ModelTrainer()
accuracy = trainer.initiate_model_trainer(train_dataset=train_dataset,
                                          val_dataset=val_dataset,
                                          test_dataset=test_dataset,
                                          root=root)
print(f">>>>>> Stage 'Model Training' completed <<<<<<\n\nx==========x")

# Print total training time
print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")