# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 02:26:20 2020

@author: Sneha
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time


def div_test_train(num,method,train_data):
    data = np.arange(0)
    while (data.size < num):
        rand_num = random.randint(0, 99)
        if(method == "train"):
            if ((rand_num in data) is False):
                data = np.append(data, rand_num)
               
        else:
            if (((rand_num in train_data) == False) and ((rand_num in data) == False)):
                data = np.append(data, rand_num)
               
    return np.sort(data) 
def generalization(method, train_num):
    e_array = mean_sq_array = time_array = time_gen_array  = w_weight_save = w_weight_save2 = w_weight_save3 = np.arange(0)
    mean_sq = 1
    for gen in range(1, 37, 2):

        weight = np.random.rand(35)  # Weights for the Network
        padding = (gen - 1) / 2
        w_zero = np.array([0])
    
        for i in range(int(padding)):  # Creating a Layer of Padding on both the Ends
            weight = np.append(w_zero, weight)
            weight = np.append(weight, w_zero)
    
        while (mean_sq > 0.01):
            if(method == "discrete"):
                mean_sq, e_array = compute_Mean_Square_discrete(weight,e_array,gen,train_num)
            elif(method == "continuous"):
                mean_sq, e_array = compute_Mean_Square_continuous(weight,e_array,gen,train_num)
        # Trying Different Values of Generalization Factor
    
        if gen == 3:       
            w_weight_save = weight
    
        elif gen == 5:       
            w_weight_save2 = weight
    
        elif gen == 9:        
            w_weight_save3 = weight
    
        time_gen_array = np.append(time_gen_array, gen)
        end = time.time()
        time_array = np.append(time_array, (end - start))
        mean_sq_array = np.append(mean_sq_array, mean_sq)
        mean_sq = 1
    return w_weight_save, w_weight_save2, w_weight_save3, mean_sq_array,time_array,time_gen_array
    
def compute_Mean_Square_discrete(weight, e_array,gen, train_num):
   
    w_value = 0.0
        # Training Phase
    for j in range(0, 70):
            q_value = j / 2

            for k in range(gen):
                w_value = w_value + weight[np.array(int(k) + int(q_value))]

            e_value = (math.cos(train_num[np.array(j)] * x)) - (w_value / gen)
            e_array = np.append(e_array, e_value)
            correction = e_value / gen  # Error Correction

            for k in range(gen):
                weight[np.array(int(k) + int(q_value))] = (weight[np.array(int(k) + int(q_value))]) + correction

            w_value = 0.0
    return np.mean(e_array ** 2), e_array

def compute_Mean_Square_continuous(weight, e_array,gen, train_num):
   
    w_value = 0.0
# Training Phase
    for j in range(0, 70):
            q_value = j/2

            for k in range(gen):           
                if (k == 0):
                    w_value = (w_value + weight[np.array(int(k) + int(q_value))] * 0.80)
                elif (k == gen-1):
                    w_value = (w_value + weight[np.array(int(k) + int(q_value))] * 0.20)
                else:
                    w_value = w_value + weight[np.array(int(k) + int(q_value))]

            e_value = (math.cos(train_num[np.array(j)] * x)) - (w_value / gen)
            e_array = np.append(e_array, e_value)
            corrected_val = e_value/gen     # Error Correction

            for k in range(gen):           
                if(k == 0):
                    weight[np.array(int(k) + int(q_value))] = (weight[np.array(int(k) + int(q_value))]) + (corrected_val * 0.80)
                elif(k == gen-1):
                    weight[np.array(int(k) + int(q_value))] = weight[np.array(int(k) + int(q_value))] + (corrected_val * 0.20)
                else:
                    weight[np.array(int(k) + int(q_value))] = weight[np.array(int(k) + int(q_value))] + corrected_val

            w_value = 0.0
    return np.mean(e_array ** 2), e_array
        
def update_weights(w_new_array,q_value, w_new, new_gen):
    w_average = w_new[np.array(int(q_value))] + w_new[np.array(int(q_value) - 1)] + w_new[np.array(int(q_value) + 1)]
    w_average = w_average / new_gen
    w_new_array = np.append(w_new_array, w_average)
    
    return w_new_array


# 100 Points for Division of Cosine Curve:
div_points = np.arange(0, 100)
# Creating a Unit Step Value:
x = 0.0628  # (2 * pi)/100 = 0.0628
y = np.arange(0)
# Initializing Variables
w_value = 0.0
w_zero = np.arange(0.0)

start = time.time()  

# Using the Sine Curve for Training
for i in range(0, 100):
    s_curve = math.cos((div_points[np.array(i)]) * x)
    y = np.append(y, s_curve)

M = np.dstack((div_points * x, y))

# Dividing the train and data as 70 &  30 respectively:

train_num = div_test_train(70,"train",None)
test_num = div_test_train(30,"train",train_num)

w_weight_save_d, w_weight_save2_d, w_weight_save3_d, mean_sq_array_d,time_array_d, time_gen_array_d = generalization("discrete",train_num)
w_weight_save_c, w_weight_save2_c, w_weight_save3_c, mean_sq_array_c,time_array_c, time_gen_array_c = generalization("continuous",train_num)


w_new_array_d = w_new_array2_d = w_new_array3_d = w_new_array_c = w_new_array2_c = w_new_array3_c = np.arange(0)
# Testing Phase
for j in range(0, 30):
    q_value = j / 2
    w_new_array_d = update_weights(w_new_array_d,q_value,w_weight_save_d[1::2], 3)
    w_new_array2_d = update_weights(w_new_array2_d,q_value,w_weight_save2_d[1::2], 5)
    w_new_array3_d = update_weights(w_new_array3_d,q_value,w_weight_save3_d[1::2], 9)
    
    w_new_array_c = update_weights(w_new_array_c,q_value,w_weight_save_c[1::2], 3)
    w_new_array2_c = update_weights(w_new_array2_c,q_value,w_weight_save2_c[1::2], 5)
    w_new_array3_c = update_weights(w_new_array3_c,q_value,w_weight_save3_c[1::2], 9)



new_test_num = test_num * x
x_tested = div_points * x

# Plotting the Output Graphs

# Gen 1
plt.figure(1)
plt.plot(new_test_num, w_new_array_d, '-y', label="Curve when Factor = 3")
plt.plot(new_test_num, w_new_array2_d, '-b', label="Curve when Factor = 5")
plt.plot(new_test_num, w_new_array3_d, '-r', label="Curve when Factor = 9")
plt.plot(x_tested, y, '-k', label="Input", linewidth=2)
plt.title("Comparison of Generalization Values - Discrete")
plt.xlabel('Radians')
plt.ylabel('cos(x)')
plt.legend(loc='upper right')

# Gen 1
plt.figure(2)
plt.plot(new_test_num, w_new_array_c, '-y', label="Curve when Factor = 3")
plt.plot(new_test_num, w_new_array2_c, '-b', label="Curve when Factor = 5")
plt.plot(new_test_num, w_new_array3_c, '-r', label="Curve when Factor = 9")
plt.plot(x_tested, y, '-k', label="Input", linewidth=2)
plt.title("Comparison of Generalization Values - Continuous")
plt.xlabel('Radians')
plt.ylabel('cos(x)')
plt.legend(loc='upper right')

plt.figure(3)
plt.plot(time_gen_array_d, mean_sq_array_d, '-k')
plt.title("Generalization vs Error - Discrete")
plt.xlabel("Generalization")
plt.ylabel("Error")

plt.figure(4)
plt.plot(time_gen_array_c, mean_sq_array_c, '-k')
plt.title("Generalization vs Error - Continuous")
plt.xlabel("Generalization - Continuous")
plt.ylabel("Error")

plt.figure(5)
plt.plot(time_gen_array_d, time_array_d, '-g')
plt.title("Generalization vs Time - Discrete")
plt.xlabel("Generalization")
plt.ylabel("Time")
plt.show()

plt.figure(6)
plt.plot(time_gen_array_c, time_array_c, '-g')
plt.title("Generalization vs Time - Continuous")
plt.xlabel("Generalization")
plt.ylabel("Time")
plt.show()
