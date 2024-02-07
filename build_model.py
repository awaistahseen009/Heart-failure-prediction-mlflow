from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import RandomizedSearchCV
import pickle

class BuildModel:
    def __init__(self, x, y, model, hyper_parameter_tuning=False, save_model=True, return_model=False, config=None):
        self.x = x
        self.y = y
        self.model = model
        self.hyper_parameter_tuning = hyper_parameter_tuning
        self.save_model = save_model
        self.return_model = return_model
        self.config = config
        self.fitted_model = None

    def build_model(self):
        if self.hyper_parameter_tuning:
            if self.config is not None:
                self.model = RandomizedSearchCV(self.model, param_distributions=self.config, cv=5, n_iter=20, verbose=True)
                self.model.fit(self.x, self.y)
        else:
            self.model.fit(self.x, self.y)

        if self.save_model:
            pickle.dump(self.model, open('model_fit.pkl', 'wb'))

        if self.return_model:
            return self.model

    def print_scores_and_params(self):
        if isinstance(self.model, RandomizedSearchCV):
            print("Model best score:", self.model.best_score_)
            print("Model best params:", self.model.best_params_)

            # Save metrics in a file
            with open('best_params.txt', 'w') as f:
                f.write("Model best score: {}\n".format(self.model.best_score_))
                f.write("Model best params: {}\n".format(self.model.best_params_))
        else:
            print("You didn't fit Random search cv, first fit that and then print these scores")

    def custom_model(self, custom_model):
        if self.hyper_parameter_tuning:
            if self.config is not None:
                custom_model = RandomizedSearchCV(custom_model, param_distributions=self.config, cv=5, n_iter=20, verbose=True)
                custom_model.fit(self.x, self.y)
        else:
            custom_model.fit(self.x, self.y)

        if self.save_model:
            pickle.dump(custom_model, open('model_fit.pkl', 'wb'))

        if self.return_model:
            return custom_model
