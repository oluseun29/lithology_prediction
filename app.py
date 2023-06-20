#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Application that predicts heart disease percentage in the population of a town
# based on the number of bikers and smokers.

# Trained on the data set of percentage of people biking
# to work each day, the percentage of people smoking, and the percentage of
# people with heart disease in an imaginary sample of 500 towns.

import streamlit as st
import numpy as np
from flask import Flask, request, render_template
import pickle

# In[2]:


# Create an app object using the Flask class.
app = get_optionFlask(__name__, template_folder='template')

# In[3]:


# Load the trained model. (Pickle file)
model = pickle.load(open('model/model.pkl', 'rb'))


# In[4]:


# Define the route to be home.
# The decorator below links the relative route of the URL to the function it is decorating.
# Here, home function is with '/', our root directory.
# Running the app sends us to index.html.
# Note that render_template means it looks for the file in the templates folder.

# use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')


# In[5]:


# You can use the methods argument of the route() decorator to handle different HTTP methods.
# GET: A GET message is send, and the server returns data
# POST: Used to send HTML form data to the server.
# Add Post method to the decorator to allow for form submission.
# Redirect to /predict page with the output
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]  # Convert string inputs to float.
    features = [np.array(int_features)]  # Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = prediction

    return render_template('index.html', prediction_text =' Formation Lithology {} '.format(output))




# In[ ]:
# When the Python interpreter reads a source file, it first defines a few special variables.
# For now, we care about the __name__ variable.
# If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
# So if we want to run our code right here, we can check if __name__ == __main__
# if so, execute it here.
# If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__": 
    app.run(port=3000, debug=True)

