from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_data():
	print("get_data")
	df = datasets.load_iris()
	return df

def holdout(df):
	print("holdout")
	X = df.data
	y = df.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	return (X_train, X_test, y_train, y_test)