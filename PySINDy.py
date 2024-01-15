#!/usr/bin/env python
# coding: utf-8

# # Applying SINDy algorithm over chaotic Lorenz system and a custom non-linear dynamical system

# In[ ]:





# In[1]:


get_ipython().system('pip install pysindy')


# In[ ]:





# In[ ]:





# # --------------------Model for Lorenz System-----------------

# The chaotic Lorenz system, also known as the **Lorenz attractor**, is a set of three coupled nonlinear ordinary differential equations proposed by Edward N. Lorenz in 1963. It is a classic example of a chaotic system and has significant implications in chaos theory and the study of nonlinear dynamics.
# The Lorenz system is given by the following equations:
#     ![image.png](attachment:image.png)
# 
# where x, y, and z are the state variables, and σ, ρ, and β are the parameters of the system. These parameters control the system's behavior and give rise to different types of dynamics.
# 
# Mathematical representation of atmospheric convection for a 2-d layer of atmospheric fluid, x is propotional to rate of convection, y is propotional to horizontal temperature variation and z is propotional to vertical temperature variation.
# ![ChessUrl](https://static.wixstatic.com/media/7a8bca_93efa84f0f9f4943ae450c596eb43c23~mv2.gif "lorenz")
# 

# ## Model-1 : SINDy over the chaotic Lorenz system's time series data

# # Problem Statement 
# **In this task, we aim to generate time series data for the Chaotic Lorenz system and then apply the pySINDy library to find the lower-dimensional governing equations of the system. The Chaotic Lorenz system is a well-known model that exhibits chaotic behavior and is often used to study complex dynamics in atmospheric convection.**

# In[243]:


import pysindy as ps
model = ps.SINDy()
model


# # Importing Libraries and Modules
# 
# 1. `numpy`:  A library for numerical computing, allowing us to handle multi-dimensional arrays and perform mathematical operations efficiently.
# 
# 2. `pandas`:  A powerful data manipulation library, useful for handling, cleaning, and transforming data.
# 
# 3. `pysindy`: A Python library for identifying governing equations from data, using sparse identification of nonlinear dynamical systems (SINDy).
# 
# 4. `scipy`: Provides the `solve_ivp` function to numerically solve ordinary differential equations(ODEs) & obtain the time-series data for the Lorenz system.
# 
# 5. `matplotlib.pyplot`: A popular plotting library, enabling us to create visualizations and graphs.
# 
# 6. `plotly.graph_objects`: An interactive plotting library, suitable for creating high-quality 3D visualizations.
# 
# 7. `Lasso` from `sklearn.linear_model`: Used for feature selection and regularization in linear models during the identification process.
# 
# 8. `PolynomialFeatures` from `sklearn.preprocessing`: Helps generate polynomial features for creating a library of candidate functions in the SINDy algorithm.

# In[259]:


import numpy as np
import pandas as pd
from pysindy import SINDy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pysindy import STLSQ
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:





# # chaotic Lorenz system time-series data generation:
# 
# 1. `lorenz_system`: This function defines the Lorenz system equations, which describe the evolution of variables `x`, `y`, and `z` over time `t`. The equations involve parameters `sigma`, `beta`, and `rho`, which control the dynamics of the system.
# 
# 2. Parameters and Initial Conditions: The code sets values for the parameters `sigma`, `rho`, and `beta`, which determine the behavior of the Lorenz system. Additionally, it specifies the initial values of `x`, `y`, and `z` using the `initial_state` variable.
# 
# 3. Time Span: The time span for integration is defined by `t_start`, `t_end`, and `num_points`, which specify the start time, end time, and the number of time points within that span.
# 
# 4. Time Integration: The `solve_ivp` function from the `scipy` library is used to numerically integrate the Lorenz system equations over the specified time span. It takes the `lorenz_system` function, initial conditions, parameters, and the desired time points `t` as inputs and returns the solution trajectory for `x`, `y`, and `z` in the `sol` variable.

# In[260]:


# Define the Lorenz system equations
def lorenz_system(t, X, sigma, beta, rho):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Set the parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Set the initial conditions
initial_state = [1.0, 1.0, 1.0]

# Define the time span
t_start = 0
t_end = 100
num_points = 10000

# Generate the time span
t = np.linspace(t_start, t_end, num_points)

# Integrate the Lorenz equations
sol = solve_ivp(lorenz_system, (t_start, t_end), initial_state, args = (sigma, beta, rho), t_eval = t)
x, y, z = sol.y


# In[ ]:





# # Data Visualization
# **This step is associated with the data visualization that shows the progress of different state variables based on time and also shows the system behaviour.**
# 
# 1. **Line Plots for Time Series Data:** The first set of plots consists of three line plots. Each plot shows the evolution of one variable (`x`, `y`, and `z`) over time (`t`). The x-axis represents time, and the y-axis represents the values of the variables. Each line is color-coded to differentiate the variables. The plots show how each variable changes over time, exhibiting the characteristic chaotic behavior of the Lorenz system.
# 
# 2. **3D Scatter Plot for the Original Data:** The code then creates a 3D scatter plot using Plotly (`go.Scatter3d`). The scatter plot displays the time series data of the Lorenz system in 3D space. The x, y, and z coordinates represent the values of the variables `x`, `y`, and `z`, respectively, at different time points. The points are plotted as markers, and their sizes are set to 2 for better visibility. The plot's layout is customized with a white background, black axes, and light grey grid lines to enhance visibility. The title of the plot is set as "Lorenz System 3D Scatter Plot."
# 
# The purpose of these plots is to visually represent the dynamics of the Chaotic Lorenz system. The line plots illustrate how the variables change over time, while the 3D scatter plot provides an intuitive visualization of the system's trajectory in the three-dimensional phase space. These visualizations are useful for understanding the chaotic behavior of the Lorenz system and gaining insights into its complex dynamics.

# In[261]:


# Plot the time series data with customized colors
fig = plt.figure(figsize=(10, 6))
plt.plot(t, x, color='blue', label='x')
plt.plot(t, y, color='orange', label='y')
plt.plot(t, z, color='green', label='z')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Chaotic Lorenz System - Time Series Data')
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 2))
plt.plot(t, x, color='black', label='x')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Chaotic Lorenz System - Time Series Data')
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 2))
plt.plot(t, y, color='orange', label='y')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Chaotic Lorenz System - Time Series Data')
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 2))
plt.plot(t, z, color='green', label='z')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Chaotic Lorenz System - Time Series Data')
plt.legend()
plt.grid(True)
plt.show()


#3-d Plot for the original data 
# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', marker=dict(size=2, color='Teal'))])
# Update the layout
fig.update_layout(
    title="Lorenz System 3D Scatter Plot",
    scene=dict(
        xaxis=dict( 
            backgroundcolor = 'white',
            color = 'black',
            gridcolor='#f0f0f0',
            title_font=dict(size=10),
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            backgroundcolor = 'white',
            color='black',
            gridcolor='#f0f0f0',
            title_font=dict(size=10),
            tickfont=dict(size=10)
        ),
        zaxis=dict(
            backgroundcolor='lightgrey',
            color='black',
            gridcolor='#f0f0f0',
            title_font=dict(size=10),
            tickfont=dict(size=10)
        )
    )
)

# Show the plot
fig.show()


# In[ ]:





# In[ ]:





# # Defining candidate library and calculating the library matrix
# 
# 1. **Coordinate Data Transformation:** The `x`, `y`, and `z` time series data obtained from the Lorenz system are converted into NumPy arrays (`np.array(x)`, `np.array(y)`, and `np.array(z)`) for further processing.
# 
# 2. **Feature Matrix Creation:** The `x`, `y`, and `z` coordinate data are combined into a single feature matrix called `coordinates` using `np.column_stack((x, y, z))`. This matrix will be used for polynomial transformation.
# 
# 3. **Polynomial Features:** A polynomial transformation is applied to the `coordinates` feature matrix to generate polynomial terms up to a specified degree. In this code, `degree = 4` is used. The `PolynomialFeatures` transformer from scikit-learn is utilized for this purpose (`poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)`).
# 
# 4. **Feature Names:** The feature names are obtained from the `PolynomialFeatures` transformer using `poly_transformer.get_feature_names(['x', 'y', 'z'])`. These names correspond to the polynomial terms created from the original `x`, `y`, and `z` coordinates.
# 
# 5. **Function Expressions and Definitions:** For each feature name, the code generates a corresponding function expression in string format. The expression replaces `^` with `**` and spaces with `*` to make it compatible with Python's evaluation. For example, the feature name `'x^2*y'` will be converted to `'x**2*y'`, and the corresponding function will be created. These functions are stored in the `functions` list as lambda functions, allowing them to take `x`, `y`, and `z` as arguments.
# 
# 6. **Library DataFrame:** The `polynomial_features` matrix is converted to a DataFrame named `library`. Each column in this DataFrame represents a polynomial term generated from the coordinates.
# 
# 7. **Adding Constant Term (Optional):** An optional step is provided to add a constant term (all 1's) to the library by uncommenting the line `#library['1'] = np.array(float(1.0))`. This term allows the model to fit an intercept in the governing equations if needed.
# 
# 8. **Assigning Column Names to Library DataFrame:** The column names of the DataFrame are updated with the corresponding feature names obtained in step 4. This step is performed using `library.columns = function_expressions`.
# 
# Overall, the code generates a library of polynomial features from the time series data of the Lorenz system. These polynomial features can be used as input data for the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm to identify the governing equations that describe the Lorenz system's behavior.

# In[293]:


# Define the coordinate data
x = np.array(x)
y = np.array(y)
z = np.array(z)

# Combine the coordinates into a single feature matrix
coordinates = np.column_stack((x, y, z))

# Define the degree of the polynomial terms
degree = 4

# Create the PolynomialFeatures transformer
poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)

# Apply the polynomial transformation
polynomial_features = poly_transformer.fit_transform(coordinates)

# Get the feature names
feature_names = poly_transformer.get_feature_names(['x', 'y', 'z'])

# Get the function expressions
function_expressions = []
functions = []
for feature in feature_names:
    expression = feature.replace('^', '**').replace(' ', '*')  # Replace '^' with '**' and space with '*' for multiplication
    function_expressions.append(expression)
    functions.append(lambda x, y, z, feature=feature: eval(expression))

#forming library as a df for the functions and polynomials
library = pd.DataFrame(polynomial_features)

#adding constant in the library
#library['1'] = np.array(float(1.0))

#assigning the library dataframe for the corresponding functions
library.columns = function_expressions
#print(function_expressions)


# In[ ]:





# In[263]:



"""
# Define custom trigonometric functions
def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

custom_library = CustomLibrary(library_functions=[sin, cos])
custom_library.fit(data)

# Step 4: Apply Sequential Threshold Least Squares (STLSQ)
model = STLSQ(threshold=0.1)
model.fit(polynomial_features, coordinates)

# Step 5: Print the coefficients
coefficients = model.coef_
print("Coefficients:")
for i, coefficient in enumerate(coefficients):
    print(f"Equation {i+1}:")
    print(coefficient)

# Create the Lasso model
lasso_model = Lasso(alpha=0.1)

# Fit the Lasso model
lasso_model.fit(library, coordinates)

# Get the coefficients
coefficients = lasso_model.coef_

# Thresholding the coefficients
threshold = 0.005  # Set your desired threshold value
coefficients[np.abs(coefficients) < threshold] = 0

# Print the coefficients
for i, coef in enumerate(coefficients):
    print(f"Coefficient {i+1}: {coef}")
"""
print(' ')


# # Applying the SINDy over the data 
# 
# 
# 1. **Coordinate Data Preparation:** The `x`, `y`, and `z` time series data are combined into a single feature matrix called `coordinates` using `np.column_stack((x, y, z))`. This matrix contains the data points of the Lorenz system in three dimensions.
# 
# 2. **SINDy Algorithm:** The SINDy algorithm is applied using the `SINDy` function from `pysindy`. The `feature_names` argument is set to `['x', 'y', 'z']`, which indicates that the governing equations should be found based on the given coordinate variables.
# 
# 3. **Fitting SINDy:** The `sindy.fit` method is called with `coordinates` as the input data and `t=t` as the time span. This step identifies the underlying dynamical equations of the Lorenz system.
# 
# 4. **Extracting Results:** The identified equations are obtained from the `sindy.equations()` method and stored in the variable `equations`. The feature library matrix used by SINDy is accessed through `sindy.feature_library` and assigned to `feature_library_matrix`. The coefficients of the candidate functions are obtained using `sindy.coefficients()` and stored in the variable `coefficients`.
# 
# 5. **Printing Results:** The code prints the library candidate functions and the sparse coefficients for the candidate functions. These are helpful in understanding the mathematical expressions used to model the Lorenz system.
# 
# 6. **Final Identified Equations:** The identified governing equations of the chaotic Lorenz system are printed using a loop that iterates through the `equations` variable, displaying each equation.
# 
# Finally the code efficiently applies the SINDy algorithm to analyze the time series data of the chaotic Lorenz system and provides the mathematical expressions for the identified governing equations. These equations offer insights into the underlying dynamics and behavior of the Lorenz system.

# In[264]:


# Calculate the library
coordinates = np.column_stack((x, y, z))

# Perform SINDy algorithm
sindy = SINDy(feature_names=['x', 'y', 'z'])
sindy.fit(coordinates, t=t)

# Get the identified equations
equations = sindy.equations()

# Get the feature library matrix created by SINDy
feature_library_matrix = sindy.feature_library
coefficients = sindy.coefficients()
# Print the feature library matrix
print("The library candidate functions")
print(feature_library_matrix.get_feature_names())
print("The sparse coefficients for the candidate functions")
print(coefficients)

# Print the identified equations
print("The final predicted governing equations of the chaotic lorenz system")
for equation in equations:
    print(equation)


# In[294]:


def lorenz_system(t, X, sigma, beta, rho):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Set the parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0 = 2.66


# # Calculating the predicted points from the predicted equations
# 
# 1. Collected the predicted the governing equations 
# 2. Now applying and predicting the points over the predicted equations

# In[265]:


# Define the Lorenz system equations based on the equations that we got after applying sindy over the data
# here we are finding and calculating the predicted points and storing them
def lorenz_system_new(t, X):
    x, y, z = X
    dx = -9.983*x + 9.982*y
    dy = 27.598*x + -0.916*y + -0.988*x*z
    dz = -2.659*z + 0.996*x*y
    return [dx, dy, dz]

# Define the parameters for the Lorenz system
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# Define the initial condition
initial_state = [1.0, 1.0, 1.0]

# Define the time span for integration
t_start = 0
t_end = 100
t = np.linspace(t_start, t_end, 10000)

# Solve the Lorenz system to obtain time series data
sol_new = solve_ivp(lorenz_system_new, (t_start, t_end), initial_state, t_eval=t)
x_new = sol_new.y[0]
y_new = sol_new.y[1]
z_new = sol_new.y[2]


# In[ ]:





# # Comparision and forecasting for the chaotic lorenz system
# 
# **3D Curves Plot for the Original and Predicted Lorenz System**
# 
# The provided code generates two 3D plots to visualize and compare the original and predicted trajectories of the Lorenz system. The first plot, labeled "Plot-1: Lorenz System 3D Time Series Curves," depicts the time evolution of the system with blue lines representing the original points and red lines representing the predicted points. This plot allows us to observe the similarities and differences between the true and forecasted trajectories in the three-dimensional space.
# 
# The second plot, labeled "Plot-2: Lorenz System 3D Scatter Plot," displays the original and predicted points as markers. The original points are shown in blue, and the predicted points are shown in red. This scatter plot provides a clearer visualization of the data points and their spatial distribution.
# 
# **Impact and Conclusion:**
# 
# The comparison between the original and predicted Lorenz system trajectories provides valuable insights into the accuracy and effectiveness of the identified governing equations obtained from the SINDy algorithm. If the predicted points closely follow the original trajectory, it indicates that the SINDy algorithm has successfully captured the underlying dynamics of the Lorenz system. On the other hand, significant deviations between the two sets of curves may indicate limitations or inaccuracies in the model.
# 
# By assessing the discrepancies between the original and predicted points, researchers can validate the SINDy algorithm's performance and determine the quality of the obtained governing equations. Furthermore, this comparison enables us to gain a deeper understanding of the system's behavior, including its sensitivity to initial conditions and the impact of parameter variations.
# 
# Overall, the 3D curves and scatter plots provide a visual representation of the Lorenz system's dynamics and its prediction accuracy using the SINDy-derived equations. These insights are essential for advancing our knowledge of chaotic systems, modeling real-world phenomena, and enhancing predictive capabilities in various fields, including physics, engineering, and climate science.

# In[266]:


#3d curves plot for the original and predicted ones for the lorenz system
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Original Points', line=dict(color='blue')))
fig.add_trace(go.Scatter3d(x=x_new, y=y_new, z=z_new, mode='lines', name='Predicted Points', line=dict(color='red')))

# Update the layout
fig.update_layout(
    title="Plot-1 : Lorenz System 3D Time Series curves",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    legend_title="Variable",
)

# Show the plot
fig.show()

#3d scatter plot for the original points and predicted points
fig = go.Figure(data=[
    go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2), name='Original Points'),
    go.Scatter3d(x=x_new, y=y_new, z=z_new, mode='markers', marker=dict(size=2), name='Predicted Points')
])

# Update the layout
fig.update_layout(
    title="Plot-2 : Lorenz System 3D Scatter Plot",
    scene=dict(
        xaxis=dict(
            backgroundcolor='white',
            color='black',
            gridcolor='#f0f0f0',
            title_font=dict(size=10),
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            backgroundcolor='white',
            color='black',
            gridcolor='#f0f0f0',
            title_font=dict(size=10),
            tickfont=dict(size=10)
        ),
        zaxis=dict(
            backgroundcolor='lightgrey',
            color='black',
            gridcolor='#f0f0f0',
            title_font=dict(size=10),
            tickfont=dict(size=10)
        )
    )
)

# Show the plot
fig.show()


# 2d plot for the original chaotic lorenz system and predicted ones
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='Original Points (x)'))
fig.add_trace(go.Scatter(x=t, y=x_new, mode='lines', name='Predicted Points (x)'))
fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Original Points (y)'))
fig.add_trace(go.Scatter(x=t, y=y_new, mode='lines', name='Predicted Points (y)'))
fig.add_trace(go.Scatter(x=t, y=z, mode='lines', name='Original Points (z)'))
fig.add_trace(go.Scatter(x=t, y=z_new, mode='lines', name='Predicted Points (z)'))

# Update the layout
fig.update_layout(
    title=" Plot-3 : Lorenz System Time Series",
    xaxis_title="Time",
    yaxis_title="Values",
    legend_title="Variable",
)

# Show the plot
fig.show()


# # Forecasting 
# 

# In[267]:


# Define the Lorenz system equations
def lorenz_system(t, X, sigma, beta, rho):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Set the parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Set the initial conditions
initial_state = [1.0, 1.0, 1.0]

# Define the time span
t_start = 150
t_end = 200
num_points = 5000

# Generate the time span
t = np.linspace(t_start, t_end, num_points)

# Integrate the Lorenz equations
sol = solve_ivp(lorenz_system, (t_start, t_end), initial_state, args=(sigma, beta, rho), t_eval=t)
x_1, y_1, z_1 = sol.y


def lorenz_system_new(t, X):
    x, y, z = X
    dx = -9.983*x + 9.982*y
    dy = 27.598*x + -0.916*y + -0.988*x*z
    dz = -2.659*z + 0.996*x*y
    return [dx, dy, dz]

# Define the initial condition
initial_state = [1.0, 1.0, 1.0]

# Define the time span for integration
# Define the time span
t_start = 150
t_end = 200
num_points = 5000
t = np.linspace(t_start, t_end, 5000)

# Solve the Lorenz system to obtain time series data
sol_new = solve_ivp(lorenz_system_new, (t_start, t_end), initial_state, t_eval=t)
x_new_1 = sol_new.y[0]
y_new_1 = sol_new.y[1]
z_new_1 = sol_new.y[2]


# In[268]:


#3d curves plot for the original and predicted ones for the lorenz system
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x_1, y=y_1, z=z_1, mode='lines', name='Original Points', line=dict(color='blue')))
fig.add_trace(go.Scatter3d(x=x_new_1, y=y_new_1, z=z_new_1, mode='lines', name='Predicted Points', line=dict(color='red')))

# Update the layout
fig.update_layout(
    title="Plot-1 : Lorenz System 3D Time Series curves",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    legend_title="Variable",
)

# Show the plot
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Model-2 :  Applying SINDy manually over a synthetic time series data

# In[204]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, median_absolute_error


# # Functions that is gonna be used in the code
# 

# In[378]:


def calculate_mean_absolute_percentage_error(true_values, predicted_values): 
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100 

def calculate_symmetric_mape(true_values, predicted_values, eps=1e-8):
    summ = ((np.abs(true_values) + np.abs(predicted_values)) + eps)
    return np.mean(np.abs(predicted_values - true_values) / summ) * 100

def print_evaluation_scores(y_true, y_pred):
    print(f"R2 score: {r2_score(y_true, y_pred)}")
    print(f"MSE score: {mean_squared_error(y_true, y_pred)}")
    print(f"MAE score: {mean_absolute_error(y_true, y_pred)}")
    print(f"Median AE score: {median_absolute_error(y_true, y_pred)}")
    print(f"MAPE score: {calculate_mean_absolute_percentage_error(y_true, y_pred)}")
    print(f"SMAPE score: {calculate_symmetric_mape(y_true, y_pred)}")


# # Non-linear system declaration

# $$
# \begin{align}
# \frac{dx_1}{dt} &= -0.5*(state1^2) - 0.8 * state2 \\
# \frac{dx_2}{dt} &= 0.8 * state1 - 0.5 * (state2^2) 
# \end{align}
# $$

# In[379]:


def nonlinear_system(y, t):
    state_var1, state_var2 = y
    dxdt = [-0.5 * (state_var1**2) - 0.8 * state_var2, 
            0.8 * state_var1 - 0.5 * (state_var2**2)]
    return dxdt


# # Data collection

# In[380]:


t_full = np.linspace(0, 15, 1501)
initial_conditions = [0.5, 0.5]

sol = odeint(nonlinear_system, initial_conditions, t_full)
sol_new = sol[:1001]

t_discover = t_full[:1001]

x1 = sol_new[:, 0]
x2 = sol_new[:, 1]


# # Plotting the original data

# In[381]:


plt.plot(t_discover, sol_new[:, 0], 'b', label='x1(t)')
plt.plot(t_discover, sol_new[:, 1], 'g', label='x2(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()


# # calculating derivatives of x1 and x2

# In[382]:


dx1dt = -0.5 * (x1**2) - 0.8 * x2
dx2dt = 0.8 * x1 - 0.5 * (x2**2)

dt = t_discover[1] - t_discover[0]
dx1dt_data = np.gradient(x1, dt)
dx2dt_data = np.gradient(x2, dt)

X = np.zeros((sol_new.shape[0], sol_new.shape[1]))
X[:, 0] = x1  # dx1/dt
X[:, 1] = x2  # dx2/dt


# # Creating polynomial candidates library

# In[384]:


from sklearn.preprocessing import PolynomialFeatures

data_df = pd.DataFrame(x1, columns=['x1'])
data_df['x2'] = x2

degree = 3

p = PolynomialFeatures(degree=degree, include_bias=True).fit(data_df)
xpoly = p.fit_transform(data_df)
new_df = pd.DataFrame(xpoly, columns=p.get_feature_names(data_df.columns))

print("Feature names:", list(new_df))  # new_df.columns.values.tolist())
print("Feature array shape:", new_df.shape)


# In[ ]:





# # Declaring lasso model and applying over x1 and x2 separately

# In[385]:


from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.0001)

new_df_train, new_df_test = new_df[:800], new_df[800:]
dx1dt_train, dx1dt_test = dx1dt[:800], dx1dt[800:]
dx2dt_train, dx2dt_test = dx2dt[:800], dx2dt[800:]

lasso_model.fit(new_df_train, dx1dt_train)
print(lasso_model.coef_)  # should give the 3rd (x2) + 4th (x1^2) argument non-zero
print(lasso_model.intercept_)
lasso_model.score(new_df_test, dx1dt_test)
fit_dx1 = pd.DataFrame(columns=new_df.columns)
fit_dx1.loc[0] = lasso_model.coef_
fit_dx1.abs().sort_values(by=0, axis=1, ascending=False)

ypred_x1 = lasso_model.predict(new_df_test)
print_evaluation_scores(dx1dt_test, ypred_x1)

# Drop features with absolute values less than 0.1
dx1_thr = fit_dx1[fit_dx1.columns[fit_dx1.abs().max() > 0.1]]
dx1_thr


# In[386]:


# PLOT results
t_test = np.linspace(8, 10, 201)

plt.plot(t_test, ypred_x1, 'r--', lw=2, label='Prediction')
plt.plot(t_discover, dx1dt, 'g', label='Actual', alpha=0.4)
plt.axvline(x=8, color='k', linestyle='--')
plt.xlabel('Time', fontsize=14)
plt.ylabel(r'$dx_1/dt$', fontsize=14)
plt.grid()
plt.legend()
plt.show()

lasso_model.fit(new_df_train, dx2dt_train)
print(lasso_model.coef_)  # should give the 2nd (x1) + last (x2^3) argument non-zero
print(lasso_model.intercept_)
lasso_model.score(new_df_test, dx2dt_test)

fit_dx2 = pd.DataFrame(columns=new_df.columns)
fit_dx2.loc[0] = lasso_model.coef_
fit_dx2.abs().sort_values(by=0, axis=1, ascending=False)

ypred_x2 = lasso_model.predict(new_df_test)
print_evaluation_scores(dx2dt_test, ypred_x2)

# Drop features with absolute values less than 0.1
dx2_thr = fit_dx2[fit_dx2.columns[fit_dx2.abs().max() > 0.1]]
dx2_thr


# In[ ]:





# In[387]:


# PLOT results
t_test = np.linspace(8, 10, 201)

plt.plot(t_test, ypred_x2, 'r--', lw=2, label='Prediction')
plt.plot(t_discover, dx2dt, 'g', label='Actual', alpha=0.4)
plt.axvline(x=8, color='k', linestyle='--')
plt.xlabel('Time', fontsize=14)
plt.ylabel(r'$dx_2/dt$', fontsize=14)
plt.grid()
plt.legend()
plt.show()


# # Forecasting model evaluation

# In[389]:


# Manually entering values for coefficients, but this can be automated
def forecast_system(y, t):
    state_var1, state_var2 = y
    dxdt = [-0.5 * (state_var1**2) - 0.8 * state_var2, 
            0.8 * state_var1 - 0.5 * (state_var2**2)]
    return dxdt

t_forecast = np.linspace(10, 15, 500)

initial_conditions_forecast = [x1[-1], x2[-1]]

sol_forecast = odeint(forecast_system, initial_conditions_forecast, t_forecast)
plt.plot(t_forecast, sol_forecast[:, 0], 'b', label='x1(t)')
plt.plot(t_forecast, sol_forecast[:, 1], 'g', label='x2(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

plt.plot(t_forecast, sol_forecast[:, 0], 'b--', label='Forecast')
plt.plot(t_full, sol[:, 0], 'g', label='Actual', alpha=0.4)
plt.axvline(x=10, color='k', linestyle='--')
plt.legend(loc='best')
plt.xlabel('Time', fontsize=14)
plt.ylabel(r'$x_1$', fontsize=14)
plt.grid()
plt.show()

plt.plot(t_forecast, sol_forecast[:, 1], 'b--', label='Forecast')
plt.plot(t_full, sol[:, 1], 'g', label='Actual', alpha=0.4)
plt.axvline(x=10, color='k', linestyle='--')
plt.legend(loc='best')
plt.xlabel('Time', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.grid()
plt.show()

true_values_x1, true_values_x2 = sol_forecast[:, 0], sol_forecast[:, 1]
predicted_values_x1, predicted_values_x2 = sol[1001:, 0], sol[1001:, 1]

for (true_values, predicted_values) in zip([true_values_x1, true_values_x2], [predicted_values_x1, predicted_values_x2]):
    print(f"R2 score: {r2_score(true_values, predicted_values)}")
    print(f"MSE score: {mean_squared_error(true_values, predicted_values)}")
    print(f"MAE score: {mean_absolute_error(true_values, predicted_values)}")
    print(f"Median AE score: {median_absolute_error(true_values, predicted_values)}")
    print(f"MAPE score: {calculate_mean_absolute_percentage_error(true_values, predicted_values)}")
    print(f"SMAPE score: {calculate_symmetric_mape(true_values, predicted_values)}")


# In[ ]:





# In[ ]:





# # ---------------------------Thanks alot__________________

# In[ ]:




