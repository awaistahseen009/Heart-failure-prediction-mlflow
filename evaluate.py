from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import mlflow
import os

class EvaluateModel:
    def __init__(self, model, x_test, y_test, experiment_name, save_metrics=False, mlflow_tracking=False):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.save_metrics = save_metrics
        self.experiment_name = experiment_name
        self.mlflow_tracking = mlflow_tracking
        self.run_id = None

    def print_evaluation_metrics(self):
        # ROC Curve
        roc_display = RocCurveDisplay.from_estimator(self.model, self.x_test, self.y_test)
        plt.title("ROC Curve - Model Evaluation")
        plt.show()

        if self.save_metrics:
            # Save ROC Curve image
            roc_display.plot()
            plt.savefig('roc_curve.png')
            plt.close()

            # Save metrics to a file
            metrics_file = 'evaluation_metrics.txt'
            with open(metrics_file, 'w') as f:
                f.write('***************Confusion Matrix***************\n')
                f.write("**********************************************\n")
                # Confusion Matrix
                y_pred = self.model.predict(self.x_test)
                f.write("Confusion Matrix:\n")
                f.write(str(confusion_matrix(self.y_test, y_pred)) + '\n')

                f.write('***************Classification Report***************\n')
                f.write("**********************************************\n")
                # Classification Report
                f.write("Classification Report:\n")
                f.write(classification_report(self.y_test, y_pred) + '\n')

                f.write('***************Precision, Recall, F1Score***************\n')
                f.write("**********************************************\n")
                # Precision, Recall, F1 Score
                f.write("Precision: {}\n".format(precision_score(self.y_test, y_pred)))
                f.write("Recall: {}\n".format(recall_score(self.y_test, y_pred)))
                f.write("F1 Score: {}\n".format(f1_score(self.y_test, y_pred)))

            print("Evaluation metrics saved to '{}'".format(metrics_file))
            print("ROC Curve image saved to 'roc_curve.png'")

        if self.mlflow_tracking:
            mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(experiment_name=self.experiment_name)
            mlflow.start_run()
            # Log parameters to the existing mlflow run
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("save_metrics", self.save_metrics)
            mlflow.log_param("mlflow_tracking", self.mlflow_tracking)

            # Log metrics to mlflow
            roc_auc = roc_auc_score(self.y_test, self.model.predict_proba(self.x_test)[:, 1])
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("precision", precision_score(self.y_test, y_pred))
            mlflow.log_metric("recall", recall_score(self.y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(self.y_test, y_pred))

            # Save the model as an artifact
            mlflow.sklearn.log_model(self.model, "model_artifact")
            mlflow.end_run()
