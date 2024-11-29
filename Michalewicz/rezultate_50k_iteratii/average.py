import re

# Define file path
file_path = "rezultate_Michalewicz30D.txt"

# Initialize lists to store the values
results = []
running_times = []

# Read and parse the file
with open(file_path, 'r') as file:
    for line in file:
        if "Rezultat" in line:
            # Extract the result value
            result = float(re.search(r"Rezultat\s*:\s*(-?[\d.]+)", line).group(1))
            results.append(result)
        #elif "Timp de rulare" in line:
        elif "Running time" in line:
            # Extract the running time value
            #time = float(re.search(r"Timp de rulare:\s*([\d.]+)", line).group(1))
            time = float(re.search(r"Running time\s*:\s*([\d.]+)", line).group(1))
            running_times.append(time)

# Calculate lowest, highest, and average values
lowest_result = min(results)
highest_result = max(results)
average_result = sum(results) / len(results)

lowest_time = min(running_times)
highest_time = max(running_times)
average_time = sum(running_times) / len(running_times)

# Display the results
print("Results Summary:")
print(f"Lowest Result: {lowest_result:.5f}")
print(f"Highest Result: {highest_result:.5f}")
print(f"Average Result: {average_result:.5f}")

print("\nRunning Time Summary:")
print(f"Lowest Running Time: {lowest_time:.5f} seconds")
print(f"Highest Running Time: {highest_time:.5f} seconds")
print(f"Average Running Time: {average_time:.5f} seconds")
