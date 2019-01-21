import random #for shuffling data
import csv #for processing CSV files
import numpy as np
import scipy.io #for reading the mat file
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if(__name__=="__main__"):
    dataset = []
    with open('iris.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row = [float(x) for x in row]
            row[2] = int(row[2])
            dataset.append(row)
    #use shuffle function with fixed seed value
    #random.seed(1)

    sample_size = [10,20,30,50]
    train_error = {10:0,20:0,30:0,50:0}
    for size in sample_size:
        new_dataset = deepcopy(dataset)
        flip = random.sample(range(len(new_dataset)),size)
        for i in flip:
            if(new_dataset[i][2] == 1):
                new_dataset[i][2] = int(random.choice([2,3]))
            if(new_dataset[i][2] == 2):
                new_dataset[i][2] = int(random.choice([1,3]))
            if(new_dataset[i][2] == 3):
                new_dataset[i][2] = int(random.choice([1,2]))
        incorrect = 0
        #plot mesh grid
        X = np.zeros(shape=(150,2))
        new_dataset_np = np.array(new_dataset)
        X[:,0] = new_dataset_np[:,0]
        X[:,1] = new_dataset_np[:,1]
        y = new_dataset_np[:,2]
        y = y -1
        h = 0.05 #step size
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h)) #2d plane

        Z = np.zeros(shape=(xx.shape[0],xx.shape[1])) #prediction labels for each xx,yy
        #predict for each point on 2D planes
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                #we have x and y value now. predict class for them using 3NN
                distance_dict = {}
                for k in range(0,150):
                    distance_dict[k] = abs(yy[i][j] - new_dataset[k][1]) + abs(xx[i][j] - new_dataset[k][0])
                distance_dict = sorted(distance_dict.items(), key=lambda kv: kv[1])
                label = [0,0,0,0] #label[0] is useless
                for l in range(0,3):
                    label[new_dataset[distance_dict[l][0]][2]] += 1
                Z[i][j] = label.index(max(label)) -1
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.ylim(yy.min(), yy.max())
        plt.title("Decision boundary for "+str(size)+" flipped number of samples")
        plt.show()
        #for finding training error
        for i in range(0,150):
            distance_dict = {}
            for j in range(0,150):
                if(i==j):
                    continue
                else:
                    distance_dict[j] = abs(new_dataset[i][1] - new_dataset[j][1]) + abs(new_dataset[i][0] - new_dataset[j][0])
            distance_dict = sorted(distance_dict.items(), key=lambda kv: kv[1])
            label = [0,0,0,0] #label[0] is useless
            for k in range(0,3):
                label[new_dataset[distance_dict[k][0]][2]] += 1
            if(label.index(max(label)) != new_dataset[i][2]):
                incorrect = incorrect + 1
        print("Error for ",size," flipped observations is ",(incorrect/150)*100)
        train_error[size] = (incorrect/150)*100
        #run kNN on this modified dataset
    x_axis = [x for x,y in train_error.items()]
    y_axis = [y for x,y in train_error.items()]
    plt.plot(x_axis,y_axis, label = "kNN error chart")
    plt.xlabel("Number of flipped data used")
    plt.ylabel("Training Error")
    plt.legend()
    plt.show()
