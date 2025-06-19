# 📊 Custom Multivariate Linear Regression Model

This repository contains a **custom-built multivariate linear regression model implemented from scratch in Python**. The model predicts **Sales** based on **TV**, **Radio**, and **Newspaper** advertising expenditures.

This project demonstrates a fundamental understanding of linear regression principles, including:
- Gradient descent for optimization
- Manual calculation of partial derivatives for the gradients

---

## 🚀 Features

- **From-Scratch Implementation**: No external machine learning libraries (like scikit-learn) were used for the core regression logic.
- **Gradient Descent**: Implements the gradient descent algorithm to iteratively optimize the model's weights and bias.
- **Mean Squared Error (MSE)**: Uses MSE as the cost function to evaluate the model's performance during training.
- **Data Visualization**: Includes a scatter plot to visualize actual vs. predicted sales, providing a clear understanding of the model's fit.

---

## 📁 Files

- `linear_regression_custom.py`: The main Python script containing the linear regression model, training loop, and plotting logic.
- `asd.csv`: The dataset used for training and testing the model. *(Ensure this file is present in the same directory as the script.)*

---

## 💻 How to Run

### 🔧 Prerequisites
Make sure you have Python installed, along with the `pandas` and `matplotlib` libraries:
```bash
pip install matplotlib pandas
```
## 📂 Dataset

Ensure you have your `asd.csv` file in the same directory as the `linear_regression_custom.py` script. The CSV should contain the following columns:

- `TV`
- `Radio`
- `Newspaper`
- `Sales`

---

## ▶️ Execute the Script
```bash
python linear_regression_model.py
```

The script will:

- Print a sample of the dataset
- Show training progress (loss at regular intervals)
- Output the final learned weights and bias
- Display a plot of actual vs. predicted sales

---

## 🧠 Model Details

The model uses the following multivariable linear equation:

Sales = w1 * TV + w2 * Radio + w3 * Newspaper + b
Where:

- `w1`, `w2`, `w3` are the weights for TV, Radio, and Newspaper
- `b` is the bias term (intercept)

The model is trained to minimize the **Mean Squared Error (MSE)**:
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

Where:

- `n` is the number of data points  
- `yᵢ` is the actual sales for the *i*-th data point  
- `ŷᵢ` is the predicted sales for the *i*-th data point

---

## 🔄 Gradient Descent

**For weights `wⱼ`:**
∂MSE/∂wⱼ = (2/n) * Σ(ŷᵢ - yᵢ) * xᵢⱼ

**For bias `b`:**
∂MSE/∂b = (2/n) * Σ(ŷᵢ - yᵢ)


---

## ⚙️ Customization

You can adjust the following parameters in `linear_regression_custom.py`:

- `learning_rate`: Controls the step size during gradient descent
- `epochs`: Number of training iterations

---

## 🤝 Contributing

Feel free to:

- Fork this repository
- Suggest improvements
- Open issues for bugs or enhancement ideas

---

## 🧑‍💻 Author

**Yashasvi Kumar Tiwari**  
Aspiring physicist, coder, and machine learning enthusiast.

Connect with me on [LinkedIn](https://www.linkedin.com/in/yashasvi-kumar-tiwari-06083a30a/)




