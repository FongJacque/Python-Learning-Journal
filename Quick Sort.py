### Jacque Jigit Fong
### Changelog
### 2 / 23 / 2025
### Version 1.1.1
### Added options for user to simulate random cases or worst case.
### 2 / 22 / 2025
### Version 1.0.1
### Ripped framework from Merge sort (Generate 1D array, Measure operation time, and main), added quick_sort function

import random
import time
import matplotlib.pyplot as plt
import numpy as np

def quick_sort(arr, low, high, worst_case=False):
    """
    This function implements the quick sort algorithm with the option to force the worst-case pivot selection.

    Parameters:
    arr (list): A list of elements (numbers) to be sorted.
    low (int): The starting index of the array/subarray to sort.
    high (int): The ending index of the array/subarray to sort.
    worst_case (bool): Flag to force worst-case pivot selection (defaults to False).

    Returns:
    list: The sorted array.
    """
    # Base case: Subarrays have at most 1 element
    if low < high:
        if worst_case:
            # Worst case: Always choose the last element as the pivot
            pivot_index = high
        else:
            pivot_index = random.randrange(low, high + 1)
        pivot_index = partition(arr, low, high, pivot_index)
        # Recursive case: Subarrays have more than 1 element
        quick_sort(arr, low, pivot_index - 1, worst_case)
        quick_sort(arr, pivot_index + 1, high, worst_case)
    return arr

def partition(arr, low, high, pivot_index):
    """
    This function partitions the array around a pivot element.

    Parameters:
    arr (list): A list of numbers.
    low (int): The starting index of the array/subarray.
    high (int): The ending index of the array/subarray.
    pivot_index (int): The index of the pivot element.

    Returns:
    int: The new pivot index after partitioning.
    """
    pivot = arr[pivot_index]
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def generate_random_1d_array(size):
    """
    This function generates a 1D array of the specified size, filled with random integers.

    Parameters:
    size (int): The size of the array to generate.

    Returns:
    list: A 1D list of random integers between 0 and 1000.
    """
    return [random.randint(0, 1000) for _ in range(size)]

def generate_sorted_array(size):
    """
    This function generates a sorted array from 0 to `size-1`.

    Parameters:
    size (int): The size of the array to generate.

    Returns:
    list: A sorted list of integers from 0 to `size-1`.
    """
    return list(range(size))

def generate_reverse_sorted_array(size):
    """
    This function generates a reverse sorted array from `size-1` down to 0.

    Parameters:
    size (int): The size of the array to generate.

    Returns:
    list: A reverse sorted list of integers from `size-1` down to 0.
    """
    return list(range(size-1, -1, -1))

def generate_array_with_same_elements(size):
    """
    This function generates an array where all elements are the same.

    Parameters:
    size (int): The size of the array to generate.

    Returns:
    list: A list where all elements are the same (e.g., all zeros).
    """
    return [random.randint(0, 1000)] * size

def measure_operation_time(arr, low, high, worst_case=False):
    """
    This function measures the total time spent to perform the quick sort operation on the given array.

    Parameters:
    arr (list): A list (1D array) of numbers to be sorted.
    low (int): The starting index of the array/subarray to sort.
    high (int): The ending index of the array/subarray to sort.
    worst_case (bool): Flag to force worst-case pivot selection.

    Returns:
    float: The time (in seconds) it took to perform the quick_sort operation.
    """
    start_time = time.time()
    quick_sort(arr, low, high, worst_case)
    end_time = time.time()
    return end_time - start_time

def simulate_multiple_runs(num_simulations, array_size, worst_case=False):
    """
    This function simulates sorting a random array multiple times and records the time for each simulation.

    Parameters:
    num_simulations (int): The number of times the sorting simulation will be repeated.
    array_size (int): The size of the array to be sorted.
    worst_case (bool): Flag to force worst-case pivot selection.

    Returns:
    list: A list of times (in seconds) for each simulation.
    """
    times = []
    for _ in range(num_simulations):
        arr = generate_random_1d_array(array_size)
        sorting_time = measure_operation_time(arr, 0, len(arr) - 1, worst_case)
        times.append(sorting_time)
    return times

def test_sorting_performance_with_simulations(num_simulations):
    """
    This function tests the performance of the quick sort algorithm by simulating sorting for multiple runs and recording the time taken for each run.
    It then outputs one graph:
    1. A red line for the average time of each array size and a best-fitting line (trend line).

    Parameters:
    None

    Returns:
    None
    """
    sizes = range(100, 2001, 100)
    all_simulation_times = []
    average_times = []

    for size in sizes:
        simulation_times = simulate_multiple_runs(num_simulations, size, worst_case=True)
        all_simulation_times.append(simulation_times)
        average_time = np.mean(simulation_times)
        average_times.append(average_time)

    # Calculates best fitting linear line
    coefficients = np.polyfit(sizes, average_times, 1)
    trend_line = np.polyval(coefficients, sizes)

    # Graphs average array size to time to sort
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, average_times, color='red', marker='o', label='Average Time')
    # Plots the best fitting linear line
    plt.plot(sizes, trend_line, color='blue', linestyle='--', label='Best Fitting Line')
    plt.xlabel("Array Size")
    plt.ylabel("Time to Sort (Seconds)")
    plt.title("Quick Sort Performance: Average Times with Best Fitting Line")
    plt.legend()
    plt.show()

def main():
    """
    Main function to run the program, where the user inputs the array size and selects the array type (random/worst-case).
    
    Parameters:
    None
    
    Returns:
    None
    """
    array_size = int(input("Enter the size of the array: "))
    array_type = input("Choose array type (random/worst-case): ").lower()
    num_simulations = int(input("Enter the number of simulations to run: "))
    if array_type == "random":
        arr = generate_random_1d_array(array_size)
        worst_case = False
    else:
        # Worst-case scenario always forces worst-case pivot
        worst_case_type = random.choice(["sorted", "reverse-sorted", "same-elements"])
        print(f"Using worst-case array type: {worst_case_type}")
        if worst_case_type == "sorted":
            arr = generate_sorted_array(array_size)
        elif worst_case_type == "reverse-sorted":
            arr = generate_reverse_sorted_array(array_size)
        elif worst_case_type == "same-elements":
            arr = generate_array_with_same_elements(array_size)
        worst_case = True
    print(f"Original Unsorted Array (first 20 elements): {arr[:20]} ...")
    sorted_array = quick_sort(arr, 0, len(arr) - 1, worst_case=worst_case)
    print(f"Sorted Array (first 20 elements): {sorted_array[:20]} ...")
    test_sorting_performance_with_simulations(num_simulations)

if __name__ == "__main__":
    main()
