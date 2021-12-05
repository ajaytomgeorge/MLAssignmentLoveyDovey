import pickle
import pandas as pd

clf = pickle.load(open('mlp_model.pkl', 'rb'))
df_test = pd.read_csv("test-i.txt", sep = " ", header = None)
test_pred = clf.predict(df_test)
test_pred = pd.DataFrame(test_pred)
test_pred
test_pred.to_csv("test-o.txt", index = False)