#@author: rymo1354
# date - 1/20/2022

def correct_ordering(arr):

    lis = [0] * len(arr)
    for i in range(len(arr)):
        lis[i] = 1
    for i in range(1, len(arr)):
        for j in range(i):
            if (arr[i] >= arr[j] and
                lis[i] < lis[j] + 1):
                lis[i] = lis[j] + 1
    maximum = 0
    for i in range(len(arr)):
        if (maximum < lis[i]):
            maximum = lis[i]
    return maximum
