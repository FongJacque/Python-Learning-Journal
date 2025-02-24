### Jacque Jigit Fong
### Changelog
### 2 / 20 / 2025
### Version 1.1.1
### Added simulations for big_o_times, producing more accurate representations of o times.
### 2 / 18 / 2025
### Version 1.0.1
### Code is functional, aditional bug testing needed
### 2/17/2025
### Version 0.2.1
### Completed measure_operation_time, run_tests, and big_o_times
### 2/16/2025
### Version 0.1.3
### Completed Search sale_by_id
### 2/15/2025
### Version 0.1.2
### Completed retrieve_latest_sale, compute_total revenue, and check_duplicate_id
### 2/14/2025
### Version 0.1.1
### Completed generate_sale_data function, utilizing file I/O, ad added chance for duplicate sales_id to generate at 5% rate
### 2/13/2025
### Version 0.0.2
### Created UI for menu
### 2/12/2025
### Version 0.0.1
### Created framework for functions defining their usage

import random
import csv
import time
import matplotlib.pyplot as plt
from datetime import date, timedelta
from datetime import datetime
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_sales_data(n, filename="sales_data.csv"):
    """
    This function generates a csv with random variables of predetermined ranges.

    Parameters:
    n (int): Variable used to generate the amount of rows.
    filename (string): Output file name.

    Returns:
    None
    """
    products = ["Widget", "Gadget", "Thingamajig", "Doohickey"]
    with open(filename, mode='w', newline='') as file:
        file_write = csv.writer(file)
        file_write.writerow(["sale_id", "sale_date", "amount", "product"])
        start_date = date(2024, 1, 1)
        sale_ids = []
        for sale_id in range(n):
            # Randomly generate some duplicate sale IDs (5%) to assist with bug testing and check for invalid entries
            if random.random() < 0.05 and sale_ids:
                sale_id = random.choice(sale_ids)
            sale_ids.append(sale_id)
            days_offset = random.randint(0, 365)
            # timedelta only modifies dates by Julian dates
            sale_date = start_date + timedelta(days=days_offset)
            amount = round(random.uniform(50, 500), 2)
            product = random.choice(products)
            file_write.writerow([sale_id, sale_date, amount, product])

def retrieve_latest_sale(filename="sales_data.csv"):
    """
    This function tracks all rows in a csv then retains the most recent sale.

    Parameters:
    filename (string): Input file to process.

    Returns:
    latest_sale (list): List of strings consisting of sale_id, sale_date, amount, product.
    """
    latest_sale = None
    latest_date = None
    seen_ids = set()
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            sale_id = int(row[0])
            # Check to skip any duplicate entries
            if sale_id in seen_ids:
                continue
            seen_ids.add(sale_id)
            sale_date = datetime.strptime(row[1], "%Y-%m-%d").date()
            amount = float(row[2])
            product = row[3]
            if latest_sale is None or sale_date > latest_date:
                latest_sale = row
                latest_date = sale_date
    return latest_sale

def compute_total_revenue(filename="sales_data.csv"):
    """
    This function tracks all rows in a csv then calculates the total revenue.

    Parameters:
    filename (string): Input file to process.

    Returns:
    total_revenue (float): Sum of all sales, float number with 2 decimal places.
    """
    total_revenue = 0.0
    seen_ids = set()
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            sale_id = int(row[0])
            if sale_id in seen_ids:
                continue
            seen_ids.add(sale_id)
            amount = float(row[2])
            total_revenue += amount
    return total_revenue

def check_duplicate_sale_ids(filename="sales_data.csv"):
    """
    This function checks all rows in a csv then returns duplicate sale ids.

    Parameters:
    filename (string): Input file to process.

    Returns:
    duplicates (list): List of integers of duplicate invalid sales ids.
    """
    duplicates = []
    seen_ids = set()
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            sale_id = int(row[0])
            if sale_id in seen_ids:
                duplicates.append(sale_id)
            else:
                seen_ids.add(sale_id)
    return duplicates

def search_sale_by_id(filename="sales_data.csv", search_id=0):
    """
    This function checks all rows in a csv which returns if requested sale id is found.

    Parameters:
    filename (string): Input file to process.
    searh_id (int): User input id to search for.

    Returns:
    row (list): A list of strings, which returns the row if sale id is found.
    none: Outputs message for user if the users' sale id is not found.
    """
    seen_ids = set()
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            sale_id = int(row[0])
            if sale_id in seen_ids:
                continue
            seen_ids.add(sale_id)
            if sale_id == search_id:
                return row
    return None

def measure_operation_time(operation, *args):
    """
    This function measures the total time spent to perform an operation.

    Parameters:
    operation (callable): Any function or method to be measured.
    *args (tuple): Variable length argument list, used to intake multiple arguments.

    Returns:
    end_time - start_time (float): time in seconds to between operation start and end time.
    """
    start_time = time.time()
    operation(*args)
    end_time = time.time()
    return end_time - start_time

def run_tests():
    """
    This function defines sizes of csv files to be operated on, formats, and outputs times to to calculate each function.

    Parameters:
    None

    Returns:
    times (dict): Names of operations as keys, and operation time listed as their values.
    """
    sizes = [100, 1000, 10000, 100000]
    times = {
        'retrieve_latest_sale': [],
        'compute_total_revenue': [],
        'check_duplicate_sale_ids': [],
        'search_sale_by_id': []
    }

    for size in sizes:
        filename = f"sales_data_{size}.csv"
        generate_sales_data(size, filename)
        times['retrieve_latest_sale'].append(measure_operation_time(retrieve_latest_sale, filename))
        times['compute_total_revenue'].append(measure_operation_time(compute_total_revenue, filename))
        times['check_duplicate_sale_ids'].append(measure_operation_time(check_duplicate_sale_ids, filename))
        times['search_sale_by_id'].append(measure_operation_time(search_sale_by_id, filename, random.randint(0, size-1)))
    return times

def simulate_performance_tests(num_simulations=1):
    """
    This function simulates multiple instances of run_test, calculates average times for each operation, and returns the individual simulation results and the best-fitting line.

    Parameters:
    num_simulations (int): The number of simulations to run, defaulted to 1.

    Returns:
    simulations (dict): A dictionary containing the individual simulation times.
    avg_times (dict): A dictionary containing the average times for each operation.
    best_fitting_lines (dict): A dictionary containing the best-fitting line (linear regression) for each operation.
    """
    sizes = [100, 1000, 10000, 100000]
    all_simulations = {
        'retrieve_latest_sale': [],
        'compute_total_revenue': [],
        'check_duplicate_sale_ids': [],
        'search_sale_by_id': []
    }

    for _ in range(num_simulations):
        times = run_tests()
        for operation in times:
            all_simulations[operation].append(times[operation])
    
    avg_times = {}
    for operation in all_simulations:
        # Contains all times of the current operation
        operation_times = all_simulations[operation]
        avg_time_for_operation = []
        for times in zip(*operation_times):
            # Calculates the average time from the current operation then adds to avg_time_for_operation
            avg_time_for_operation.append(np.mean(times))
        avg_times[operation] = avg_time_for_operation

    # Perform linear regression to fit the best-fitting line for each operation
    best_fitting_lines = {}
    slopes = {}

    for operation, times in avg_times.items():
        # Log scale on X and Y axises for better fit lines
        X = np.log10([100, 1000, 10000, 100000])
        y = np.log10(times)
        # stats.linregress() only needs slope and intercept, so we void the r-value, p-value, and stderr
        slope, intercept, _, _, _ = stats.linregress(X, y)
        slopes[operation] = slope
        best_fitting_lines[operation] = slope * X + intercept

    return all_simulations, avg_times, best_fitting_lines, slopes

def big_o_times_with_simulations():
    """
    This function simulates the performance tests multiple times, calculates the average times,
    and graphs the individual simulations along with the best fitting line.

    Parameters:
    None

    Returns:
    None
    """
    # Ask the user for the number of simulations
    num_simulations = int(input("Enter the number of simulations to run: "))

    print(f"\nRunning Performance Tests with {num_simulations} simulations...")
    all_simulations, avg_times, best_fitting_lines, slopes = simulate_performance_tests(num_simulations)

    sizes = [100, 1000, 10000, 100000]

    # Output message for each operation, to calculate average time depending on number of simulations.
    for operation, avg_time in avg_times.items():
        print(f"\n{operation}:")
        for size, avg in zip(sizes, avg_time):
            print(f"  Size {size}: {avg:.9f} seconds")
    colors = {
        'retrieve_latest_sale': 'blue',
        'compute_total_revenue': 'green',
        'check_duplicate_sale_ids': 'red',
        'search_sale_by_id': 'orange'
    }

    # First graph: individual simulations with different colors for each operation
    plt.figure(figsize=(10, 6))
    for operation, simulations in all_simulations.items():
        for simulation in simulations:
            plt.plot(sizes, simulation, color=colors[operation], alpha=0.4)
    plt.title('Individual Simulations of Operations on Sales Data')
    plt.xlabel('Dataset Size')
    plt.ylabel('Time Taken (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # Second graph: Average times to size of data set and corresponding best fitting lines
    plt.figure(figsize=(10, 6))
    for operation, avg_time in avg_times.items():
        plt.plot(sizes, avg_time, label=f"{operation} (Avg.)", marker='o', color=colors[operation], linestyle='-', linewidth=2)
        plt.plot(sizes, 10 ** best_fitting_lines[operation], label=f"{operation} (Best Fit)", color=colors[operation], linestyle='--', linewidth=2)
        # Output value of best fitting line, helping to determine most efficient times
        print(f"Slope of best-fitting line for '{operation}': {slopes[operation]:.4f}")
    plt.title('Average Times and Best Fitting Line of Operations on Sales Data')
    plt.xlabel('Dataset Size (Number of Records)')
    plt.ylabel('Time Taken (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def main_menu():
    """
    This function displays a menu for the user to interact with and choose an option.

    Parameters:
    None

    Returns:
    None
    """
    while True:
        print("\n--- Sales Data Operations Menu ---")
        print("1. Generate Sales Data")
        print("2. Retrieve the Latest Sale")
        print("3. Compute Total Revenue")
        print("4. Check for Duplicate Sale IDs")
        print("5. Search Sale by ID")
        print("6. Run Performance Tests (Big-O Time Analysis)")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            n = int(input("Enter the number of sales records to generate: "))
            filename = "sales_data.csv"
            generate_sales_data(n, filename)
            print(f"\nSales data has been generated and saved to {filename}.")

        elif choice == '2':
            filename = "sales_data.csv"
            latest_sale = retrieve_latest_sale(filename)
            if latest_sale:
                print("\nLatest Sale:", latest_sale)
            else:
                print("\nNo sales data available.")

        elif choice == '3':
            filename = "sales_data.csv"
            total_revenue = compute_total_revenue(filename)
            print(f"\nTotal Revenue: ${total_revenue:.2f}")

        elif choice == '4':
            filename = "sales_data.csv"
            duplicates = check_duplicate_sale_ids(filename)
            if duplicates:
                print(f"\nDuplicate Sale IDs: {duplicates}")
            else:
                print("\nNo duplicate Sale IDs found.")

        elif choice == '5':
            filename = "sales_data.csv"
            search_id = int(input("Enter the sale ID to search for: "))
            sale = search_sale_by_id(filename, search_id)
            if sale:
                print(f"\nSale Found: {sale}")
            else:
                print(f"\nNo sale found with ID {search_id}.")

        elif choice == '6':
            big_o_times_with_simulations()

        elif choice == '7':
            print("Exitting program.\n")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

if __name__ == "__main__":
    main_menu()