# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



# %%
# Load the dataset
data = pd.read_csv("data/Tesla_Stock.csv")
print(data.head())

# %%
data.shape 

# %%
data.describe()

# %%
data.isnull().sum()

# %%
data.dtypes

# %%
data.memory_usage()

# %%
X = data[['Open','High','Low','Volume']]
y = data['Close']

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
model = LinearRegression()

model.fit(X_train, y_train)

# %%
predictions = model.predict(X_test)

print(predictions[:10])

# %%
mse = mean_squared_error(
    y_test,
    predictions
)

print("MSE:", mse)

# %%
r2 = r2_score(
    y_test,
    predictions
)

print("R2 Score:", r2)
#Generate Correlation Matrix

correlation_matrix = data.corr(numeric_only=True)

print(correlation_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')
plt.show()

# %%
plt.figure(figsize=(10,5))

plt.plot(
    y_test.values,
    label="Actual"
)

plt.plot(
    predictions,
    label="Predicted"
)

plt.xlabel("Days")
plt.ylabel("Stock Price")

plt.title("Actual vs Predicted Prices")

plt.legend()

plt.show()

plt.savefig("prediction_graph.png")

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')

plt.savefig('correlation_heatmap.png')

plt.show()
