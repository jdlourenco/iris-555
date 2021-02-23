from iris.data import get_data, holdout
from iris.pipeline import TaxiFarePipeline
import joblib

class Trainer:

	def __init__(self):
		pass

	def fit(self):
		print("fitting")
		self.pipeline.fit(self.X_train, self.y_train)

	def save(self):
		print("save")
		joblib.dump(self.pipeline, f'pipeline.joblib')

	def evaluate(self):
		pass
		# log_metric
		# metric = compute_metric(y_pred, self.y_test)
		# print(metric)
		# return metric

	def train(self):
		print("let's train")
		# get_data
		df = get_data()
		# holdout
		self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)
		# create pipeline
		print("let's get the pipeline")
		self.pipeline = TaxiFarePipeline().get_pipeline()
		# fit pipeline
		fitted_pipeline = self.fit()
		# save to disk
		self.save()

if __name__ == "__main__":
	trainer = Trainer()
	pipeline = trainer.train()
