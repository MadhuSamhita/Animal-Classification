Animal Classification Project: From Data Preparation to Model Evaluation

Data Collection:
Download the Zoo dataset and Class dataset from Kaggle. The Zoo dataset contains information about animals, while the Class dataset contains class information.

Data Merging:
Merge the Zoo dataset and Class dataset based on the common attributes: Class_number in Class dataset and class_type in Zoo dataset.

Data Preprocessing:
Remove irrelevant columns that won't contribute to the classification task.

Data Visualization:
Create a correlation matrix to explore the relationships between attributes and identify potential patterns.
Generate a pie chart to visualize the population ratios of all 7 classes of animals.

Handling Imbalance:
Since the dataset is small and imbalanced, perform oversampling to balance the class distribution. This can improve model performance.

Train-Test Split:
Split the preprocessed and balanced dataset into training and testing sets.

Model Building - Random Forest Classifier:
Choose a Random Forest Classifier as the classification algorithm due to its robustness and ability to handle various types of data.
Train the Random Forest model on the training data.

Model Testing and Evaluation:
Test the trained model on the testing dataset to assess its predictive performance.
Compute the accuracy of the model to determine how well it predicts animal classes.
Generate a confusion matrix to show the counts of True Positive, True Negative, False Positive, and False Negative predictions.
Genearte a classification report that gives accuracy score, precision, recall and f1 score.
Evaluate the model by observing the predictions it makes on run time data i.e, input given after tetsing by the user.
