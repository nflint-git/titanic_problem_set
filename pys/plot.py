import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

# string to list to allow for easy input into a function
def string_to_list(input_string):
    return input_string.split()


#countplot function that allows inserted df, columns, and hue to be arguments with the very latter being optional
def count_plot(df, columns, hue=None):
    df = df.reset_index(drop=True)
    columns = string_to_list(columns)
    for column in columns:
        if df[column].dtype != 'category':
            df[column] = df[column].astype('category')
    num_rows = math.ceil(len(columns)/2)
    f, axes = plt.subplots(num_rows, 2, figsize=(20, 10))
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()  # Multiple axes (flatten into 1D array)
    else:
        axes = [axes]  # Single axis (wrap in list)

    for i, column in enumerate(columns):
        sns.countplot(x=column, data=df, hue=hue, ax=axes[i])
        axes[i].set_title(f"Countplot of {column}")
        axes[i].grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")

    for j in range(i + 1, len(axes)):
        f.delaxes(axes[j])
    plt.tight_layout()
    plt.show()  

#distribution plot
def dist_plot(df, columns, hue=None):
    df = df.reset_index(drop=True)
    columns = string_to_list(columns)
    num_rows = math.ceil(len(columns)/2)
    f, axes = plt.subplots(num_rows, 2, figsize=(20, 10))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()  # Multiple axes (flatten into 1D array)
    else:
        axes = [axes]  # Single axis (wrap in list)

    for i, column in enumerate(columns):
        sns.histplot(x=column, data=df, hue=hue, ax=axes[i])
        axes[i].set_title(f"Distplot of {column}")
        axes[i].grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")

    for j in range(i + 1, len(axes)):
        f.delaxes(axes[j])
    plt.tight_layout()
    plt.show()  


