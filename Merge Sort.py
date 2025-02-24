### Jacque Jigit Fong
### Changelog
### 2 / 18 / 2025
### Version 1.0.1
### Functional program takes user input for size, adjusted to simulate multiple occurrences to create new best fitting lines
### Version 0.0.1
### Implemented merge sort, user input for array size, and measure operation time.

import random
import time
import matplotlib.pyplot as plt
import numpy as np

def merge_sort(arr):
    """
    This function implements the merge sort algorithm to sort a given array.

    Parameters:
    arr (list): A list of elements (numbers) to be sorted.

    Returns:
    list: A sorted version of the input array.
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr

def generate_random_1d_array(size):
    """
    This function generates a 1D array of the specified size, filled with random integers.

    Parameters:
    size (int): The size of the array to generate.

    Returns:
    list: A 1D list of random integers between 0 and 1000.
    """
    random_array = []
    for _ in range(size):
        random_number = random.randint(0, 1000)
        random_array.append(random_number)
    return random_array

def measure_operation_time(arr):
    """
    This function measures the total time spent to perform the merge sort operation on the given array.

    Parameters:
    arr (list): A list (1D array) of numbers to be sorted.

    Returns:
    float: The time (in seconds) it took to perform the merge_sort operation.
    """
    start_time = time.time()
    merge_sort(arr)
    end_time = time.time()
    return end_time - start_time

def simulate_multiple_runs(num_simulations, array_size):
    """
    This function simulates sorting a random array multiple times and records the time for each simulation.

    Parameters:
    num_simulations (int): The number of times the sorting simulation will be repeated.
    array_size (int): The size of the array to be sorted.

    Returns:
    list: A list of times (in seconds) for each simulation.
    """
    times = []
    for _ in range(num_simulations):
        arr = generate_random_1d_array(array_size)
        sorting_time = measure_operation_time(arr)
        times.append(sorting_time)
    return times

def test_sorting_performance_with_simulations():
    """
    This function tests the performance of the merge sort algorithm by simulating sorting for multiple runs and recording the time taken for each run.
    It then outputs one graph:
    1. A red line for the average time of each array size and a best-fitting line (trend line).

    Parameters:
    None

    Returns:
    None
    """
    num_simulations = int(input("Enter the number of simulations: "))
    sizes = range(100, 2001, 100)
    all_simulation_times = []
    average_times = []
    for size in sizes:
        simulation_times = simulate_multiple_runs(num_simulations, size)
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
    plt.title("Merge Sort Performance: Average Times with Best Fitting Line")
    plt.legend()
    plt.show()

def main():
    """
    Main function to run the program, where the user inputs the array size, the program generates a random array, sorts it, and outputs the original array and the sorted array.

    Parameters:
    None

    Returns:
    None
    """
    array_size = int(input("Enter the size of the array: "))
    arr = generate_random_1d_array(array_size)
    original_array = arr.copy()
    sorting_time = measure_operation_time(arr)
    # Only how first 20 elements for readability
    print(f"Original Array (first 20 elements): {original_array[:20]} ...")
    print(f"Sorted Array (first 20 elements): {arr[:20]} ...")
    print(f"\nTime to sort an array of size {array_size}: {sorting_time:.9f} seconds")
    test_sorting_performance_with_simulations()

if __name__ == "__main__":
    main()
