import numpy as np
import matplotlib.pyplot as plt

import math

def rotate_vector(vector, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)
    
    # Extract components of the vector
    x, y = vector
    
    # Calculate the new components after rotation
    new_x = x * math.cos(angle_radians) - y * math.sin(angle_radians)
    new_y = x * math.sin(angle_radians) + y * math.cos(angle_radians)
    
    # Return the rotated vector
    return (new_x, new_y)


def find_points_on_line(A, B, shorter_distance):
    x1, y1 = A
    x2, y2 = B
    
    # Calculate the distance between A and B
    AB = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Calculate the new shorter distance between C and D
    CD = AB - shorter_distance
    
    # Calculate the ratio of CD to AB
    m = CD / AB
    
    # Calculate coordinates of points C and D
    x_c = x1 + m * (x2 - x1)
    y_c = y1 + m * (y2 - y1)
    
    x_d = x2 - m * (x2 - x1)
    y_d = y2 - m * (y2 - y1)
    
    return (x_c, y_c), (x_d, y_d)
# x, y = 3, 0

# x, y = rotate_vector((x, y), 22.5)
# print("x = ", x, "y = ", y)



print(find_points_on_line((2.77163859753386, 1.1480502970952693), (1.1480502970952693, 2.77163859753386), 0.9))


