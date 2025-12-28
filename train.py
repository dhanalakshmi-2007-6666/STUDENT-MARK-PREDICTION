import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
data = pd.read_csv("Student_Performance.csv")
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
x = data[['Hours Studied',
          'Previous Scores',
          'Extracurricular Activities',
          'Sleep Hours',
          'Sample Question Papers Practiced']]
y = data['Performance Index']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = r2_score(y_test, y_pred)
print("Model accuracy:", accuracy)
plt.scatter(x_test['Hours Studied'], y_test)
plt.plot(x_test['Hours Studied'], y_pred)
plt.xlabel("Hours Studied")
plt.ylabel("Performance Index")
plt.title("Student Performance Prediction")
plt.show()
pickle.dump(model, open("student.pkl", "wb"))