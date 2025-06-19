import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("asd.csv")

print("Smaple Data")
print(data.head())

weights={
    "TV":0.0,
    "Radio":0.0,
    "Newspaper":0.0
}

bias=0.0

def predict(row,weights,bias):
    result=0

    for feature in weights:
        result+=weights[feature]*row[feature]
    return result+bias

    #this retuns w1*x1+w2*x2+w3*x3+b

def computer_loss(data,weights,bias):
    total_error=0.0
    n=len(data)

    for _,row in data.iterrows():
        predicted=predict(row,weights,bias)
        actual=row["Sales"]
        total_error+=(actual-predicted)**2
    
    return total_error/n

    #this return 1/n* summation i=0 to n (yi-(w1*x1+w2*x2+w3*x4+b))**2

def gradient_descent(data,weights,bias,learning_rate):
    n=len(data)
    weight_grads={feature:0.0 for feature in weights}
    bias_grad=0.0

    for _,row in data.iterrows():
        prediction=predict(row,weights,bias)
        actual=row["Sales"]
        error=prediction-actual

        for feature in weights:
            weight_grads[feature]+=(2/n)*error*row[feature]
        bias_grad+=(2/n)*error

    for feature in weights:
        weights[feature]-=learning_rate*weight_grads[feature]
    bias-=learning_rate*bias_grad

    return weights,bias

learning_rate=0.000001
epochs=3000

print("\nTraining Model....\n")

for epoch in range(epochs):
    weights,bias=gradient_descent(data,weights,bias,learning_rate)

    if(epoch%50==0):
        loss=computer_loss(data,weights,bias)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal weights and bias:")
for feature in weights:
    print(f"{feature}: {weights[feature]:.4f}")
print(f"Bias: {bias:.4f}")

actual_sales=[]
predicted_sales=[]

for _,row in data.iterrows():
    actual_sales.append(row["Sales"])
    predicted_sales.append(predict(row,weights,bias))

plt.scatter(actual_sales,predicted_sales,color="blue")
plt.plot([min(actual_sales),max(actual_sales)],[min(predicted_sales),max(predicted_sales)],color="red",linestyle="--")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Custom Linear Regression)")
plt.grid(True)
plt.show()