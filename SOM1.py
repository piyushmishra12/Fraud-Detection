import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
# x, y are the grid dimensions
# input_len is the number of features that we have in X, ie, 14 + 1 = 15
# sigma is the radius of the circle, 1.0 being the default

# Assigning random weights
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualising the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
# From the legend it is clear that the black coloured blocks have minimum
# Euclidean distance from the centre meaning the lighter the colour gets
# the chance of being a Fraud is higher

# However, what matters here is the fact that we need to identify those customers
# who got approved and are likely to cheat instead of those customers who did
# not get approved and are likely to cheat.
# Approved or not is in y.
markers = ['o', 's']
colours = ['b', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,  # plotting at the
         w[1] + 0.5,  # centre of the block
         markers[y[i]],
         markeredgecolor=colours[y[i]],
         markerfacecolor='None',
         markersize=5,
         markeredgewidth=1)
show()
# y[i] has only 2 values 0 or 1, so if y[i] = 0, marker will be o and if 1, marker will be s
# [i] has only 2 values 0 or 1, so if y[i] = 0, colour will be r and if 1, colour will be s

mappings = som.win_map(X)
# From the SOM we can see the lightest colours in the grid are at (4, 3)
frauds = mappings[(4, 3)]
# The data in 'frauds' is the scaled data
frauds = sc.inverse_transform(frauds)
