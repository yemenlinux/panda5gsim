# functions to use with dataset analysis

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 

def filter_data(data, by_columns, wanted_columns, k_v=None):
    where = np.where(np.ones(len(data))==1, 1,0)
    if k_v:
        for k, v in k_v.items():
            where = np.where(data[k] == v , 1,0) & where
    #
    filtered = data.loc[where==1].groupby(by_columns)[wanted_columns].mean()
    filtered = filtered.reset_index()
    return filtered

def polyfit_prob(x, y, new_x, deg = 3, clip='no'):
    poly = np.polyfit(x, y, deg)
    if clip.lower() == 'no':
        return np.poly1d(poly)(new_x)
    return np.clip(np.poly1d(poly)(new_x),0,1)

def generate_coordinate_matrix(n, m):
    """
    Generates an n*m matrix containing coordinates of points (x,y) within 0-n and 0-m.

    Args:
      n: Number of rows (height) in the matrix.
      m: Number of columns (width) in the matrix.

    Returns:
      A numpy array of shape (n, m, 2) containing x and y coordinates.
    """
    x, y = np.meshgrid(np.arange(n), np.arange(m))
    return x, y, np.stack((x, y), axis=-1)

def apply_function_to_x(matrix, func):
    """
    Applies a function to the x coordinates of a coordinate matrix.

    Args:
      matrix: A numpy array of shape (n, m, 2) containing x and y coordinates.
      func: The function to apply to the x coordinates.

    Returns:
      A new matrix with the modified x coordinates.
    """
    return np.stack((func(matrix[:, :, 0]), matrix[:, :, 1]), axis=-1)

def apply_function_to_y(matrix, func):
    """
    Applies a function to the y coordinates of a coordinate matrix.

    Args:
      matrix: A numpy array of shape (n, m, 2) containing x and y coordinates.
      func: The function to apply to the x coordinates.

    Returns:
      A new matrix with the modified x coordinates.
    """
    return np.stack((matrix[:, :, 0], func(matrix[:, :, 1]) ), axis=-1)

def matrix_from_data(data, xkey, ykey, zkey, xnew = 1000, ynew = 1000, k_v_filter = None, method = 'nearest'):
    """ iterpolate x,y in a 3D data to provide grid matrix
    Arg:
        data: pandas dataframe
        xkey: the x column
        ykey: the y column
        zkey: the z column (the metric)
        xnew: number of points in x direction, default = 1000
        ynew: number of points in y direction, default = 1000
        k_v_filter: dictionary of conditions to filter the dataset
        method: the interpolation method, available: (nearest', 'linear', 'cubic')
                defult = 'nearest'
    Returns:
        retun 3 matrix X, Y, Z
        X is the x coordinat
        Y is y coordinat
        Z is the probability of a metric key in a dataset
        
    Example:
        kv = {'Environment': 'Urban', 'RayLoS': 1}
        X, Y, Z = matrix_from_data(data, 'd2D', 'h', 'RayLoS', k_v_filter = kv, method = 'linear')
    """
    #
    num_records = data.shape[0]
    where = np.where(np.ones(len(data)) == 1, 1,0) 
    if k_v_filter:
        for k,v in k_v_filter.items():
            where = np.where(data[k] == v, 1,0) & where
    #
    #
    x = np.linspace(0, xnew, xnew)
    y = np.linspace(0, ynew, ynew)
    X, Y = np.meshgrid(x, y)
    #
    data = data.loc[where == 1].groupby([xkey, ykey])[[zkey]].count()
    data = data.reset_index()
    data[zkey] = data[zkey]/num_records
    #
    Z = griddata(data[[xkey, ykey]], data[zkey], (X, Y), method=method)
    return X, Y, Z

def plot_matrix_from_data(data, 
                          xkey, 
                          ykey, 
                          zkey, 
                          col= 'Environment', 
                          xnew = 1000, 
                          ynew = 1000, 
                          k_v_filter = None, 
                          method = 'nearest',
                         xlabel = '$d_{2D}$ [m]',
                         ylabel = '$(h_{tx} - h_{rx})$ [m]',
                         zlabel = '$P_{LoS}$'):
    """ iterpolate x,y in a 3D data to provide grid matrix
    Arg:
        data: pandas dataframe
        xkey: the x column
        ykey: the y column
        zkey: the z column (the metric)
        xnew: number of points in x direction, default = 1000
        ynew: number of points in y direction, default = 1000
        k_v_filter: dictionary of conditions to filter the dataset
        method: the interpolation method, available: (nearest', 'linear', 'cubic')
                defult = 'nearest'
    Returns:
        retun 3 surface plot
        
    Example:
        kv = {'RayLoS': 1}
        fig = plot_matrix_from_data(data, 'd2D', 'h', 'RayLoS', k_v_filter = kv, method = 'nearest')
    """
    WIDTH_SIZE = 30
    HEIGHT_SIZE = 6.5
    fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
    l = len(data[col].unique())
    for i, env in enumerate(data[col].unique()):
        nfig = int(f'1{l}{i+1}')
        ax = fig.add_subplot(nfig, projection='3d',  title=f'{env}')
        X, Y, Z = matrix_from_data(data, 
                                xkey, 
                                ykey, 
                                zkey, 
                                xnew = xnew, 
                                ynew = ynew, 
                                k_v_filter = k_v_filter, 
                                method = method)
        #
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        # ax.set_zlim(0,1)
        ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
        

def image_matrix_from_data(data, 
                          xkey, 
                          ykey, 
                          zkey, 
                          col= 'Environment', 
                          xnew = 1000, 
                          ynew = 1000, 
                          k_v_filter = None, 
                          method = 'nearest',
                         xlabel = '$d_{2D}$ [m]',
                         ylabel = '$(h_{tx} - h_{rx})$ [m]',
                         ):
    """ iterpolate x,y in a 3D data to provide grid matrix
    Arg:
        data: pandas dataframe
        xkey: the x column
        ykey: the y column
        zkey: the z column (the metric)
        xnew: number of points in x direction, default = 1000
        ynew: number of points in y direction, default = 1000
        k_v_filter: dictionary of conditions to filter the dataset
        method: the interpolation method, available: (nearest', 'linear', 'cubic')
                defult = 'nearest'
    Returns:
        retun 3 surface plot
        
    Example:
        kv = {'RayLoS': 1}
        fig = plot_matrix_from_data(data, 'd2D', 'h', 'RayLoS', k_v_filter = kv, method = 'nearest')
    """
    WIDTH_SIZE = 30
    HEIGHT_SIZE = 6.5
    fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
    l = len(data[col].unique())
    for i, env in enumerate(data[col].unique()):
        nfig = int(f'1{l}{i+1}')
        ax = fig.add_subplot(nfig,  title=f'{env}')
        X, Y, Z = matrix_from_data(data, 
                                xkey, 
                                ykey, 
                                zkey, 
                                xnew = xnew, 
                                ynew = ynew, 
                                k_v_filter = k_v_filter, 
                                method = method)
        #
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.imshow(Z, cmap=plt.cm.coolwarm)
    
def plot_line_from_data(data, 
                          xkey, 
                          ykey, 
                          zkey, 
                          col= 'Environment', 
                          xnew = 1000, 
                          ynew = 1000, 
                          k_v_filter = None, 
                          method = 'nearest',
                         xlabel = '$d_{2D}$, $\\Delta h$ [m]',
                         ylabel = '$P_{LoS}$]'):
    """ iterpolate x,y in a 3D data to provide grid matrix and plot lines
    Arg:
        data: pandas dataframe
        xkey: the x column
        ykey: the y column
        zkey: the z column (the metric)
        xnew: number of points in x direction, default = 1000
        ynew: number of points in y direction, default = 1000
        k_v_filter: dictionary of conditions to filter the dataset
        method: the interpolation method, available: (nearest', 'linear', 'cubic')
                defult = 'nearest'
    Returns:
        retun 3 surface plot
        
    Example:
        kv = {'RayLoS': 1}
        fig = plot_matrix_from_data(data, 'd2D', 'h', 'RayLoS', k_v_filter = kv, method = 'nearest')
    """
    WIDTH_SIZE = 30
    HEIGHT_SIZE = 6.5
    fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
    l = len(data[col].unique())
    for i, env in enumerate(data[col].unique()):
        nfig = int(f'1{l}{i+1}')
        ax = fig.add_subplot(nfig, projection='3d',  title=f'{env}')
        X, Y, Z = matrix_from_data(data, 
                                xkey, 
                                ykey, 
                                zkey, 
                                xnew = xnew, 
                                ynew = ynew, 
                                k_v_filter = k_v_filter, 
                                method = method)
        # line
        x = X[0,:]
        y = Y[:,0]
        mu = Z.mean()
        mean_d = Z.mean(axis = 0)
        fit_d = polyfit_prob(x, mean_d, x)
        mean_h = Z.mean(axis = 1)
        fit_h = polyfit_prob(y, mean_h, x)
        ax = fig.add_subplot(nfig, title=f'{env}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x, fit_d,  label = 'd')
        ax.plot(y, fit_h,  label = 'h')
        ax.legend()
        
def plot_func(func, x, xlabel = '', ylabel = '', names = None):
    WIDTH_SIZE = 30
    HEIGHT_SIZE = 6.5
    environments = [('High-rise Urban', 0.5, 300, 50),
                    ('Dense Urban', 0.5, 300, 20),
                    ('Urban', 0.3, 500, 15),
                    ('Suburban', 0.1, 750, 8)]
    fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
    for f,t in enumerate(environments):
        env, alpha, beta, gamma = t
        nfig = int(f'14{f+1}')
        ax = fig.add_subplot(nfig, title=f'{env}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if isinstance(func, list):
            for i,f in enumerate(func):
                y = f(x, alpha, beta, gamma)
                if names:
                    ax.plot(x, y, label = f'{names[i]}')
                else:
                    ax.plot(x, y, label = f'{i}')
        else:
            y = func(x, alpha, beta, gamma)
            ax.plot(x, y, label = 'test')
        ax.legend()
        

