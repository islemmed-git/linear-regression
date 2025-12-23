import pandas as pd
import matplotlib.pyplot as plt 

# Load dataset from CSV file
data = pd.read_csv('data.csv')

# Mean Squared Error loss function
def loss_function(m, b, points):
    total_error = 0

    # Loop through all data points
    for i in range(len(points)): 
        x = points.iloc[i].studytime   # input feature
        y = points.iloc[i].score       # target value

        # Squared error
        total_error += (y - (m * x + b)) ** 2 

    # Return average error
    return total_error / float(len(points))


# Gradient Descent function
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    # Compute gradients
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        # Partial derivative with respect to m
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now)) 

        # Partial derivative with respect to b
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    #Update parameters
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L 

    return m, b


#Initial values
m = 0              # slope
b = 0              # intercept
L = 0.0001         # learning rate
epochs = 300       # number of iterations

# Training loop
for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

# Print final parameters
print(m, b)

# Plot data points
plt.scatter(data.studytime, data.score, color='black')

# Plot regression line
plt.plot(
    list(range(20, 100)),
    [m * x + b for x in range(20, 100)],
    color='green'
)

plt.show()

