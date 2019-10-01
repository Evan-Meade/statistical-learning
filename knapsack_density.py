'''
knapsack_density.py
Evan Meade, 2019

This script implements a rapid value density solution to the knapsack problem.

The knapsack problem is an optimization problem where there are a number of
items with certain values and weights. However, the knapsack can only carry
a certain total weight. How does one decide what to pack to maximize value
without exceeding the maximum weight?

The approach taken here is to compute the "value density" of each item;
essentially, this is the object's value divided by its cost, or "weight."
Then, the algorithm adds objects in order of value density until the
maximum weight is reached.

Script Function:
This script is designed to automate the process of determining a packing list
for the knapsack problem according to the value density model. It includes
procedures for reading in the .dat file and outputing the algorithm's results.

Dataset Format:
At the moment, this script assumes only one variable each for value and cost.
Each line of the .dat file should be structured as:
{value} {cost}
All values must be quantitative, and will be processed as floats.

Execution Format:
python knapsack_density.py {items.dat} {max weight}

'''

import sys

import dat_operations as dato


def main():
    # Read in execution arguments
    file_name = sys.argv[1]
    max_cost = float(sys.argv[2])

    # Breaks down data file into value and cost vectors
    indices = [1, 2]
    [value, cost, extra] = dato.slice_data(dato.read_dat(file_name), indices)

    # Computes packing list with most value dense items
    [packing_list, total_value, total_cost] = optimize_knapsack(value, cost, max_cost)

    # Prints results to console
    print(f"\nLocally optimal packing includes objects at indices:\n")
    print(packing_list)
    print(f"\nThis gives a total value of {total_value} for a total cost of {total_cost}\n")


def optimize_knapsack(value, cost, max_cost):
    total_value = 0
    total_cost = 0
    packing_list = []

    density = value_density(value, cost)
    indices = range(0, len(density))
    sorted_density = [x for _, x in sorted(zip(density, indices), reverse=True)]
    print(sorted_density)

    for i in range(0, len(sorted_density)):
        index = sorted_density[i]

        if total_cost + cost[index] <= max_cost:
            packing_list.append(index)

            total_value += value[index]
            total_cost += cost[index]

    return [packing_list, total_value, total_cost]


def value_density(value, cost):
    density = []

    for i in range(0, len(value)):
        density.append(value[i] / cost[i])

    return density


if __name__ == '__main__':
    main()
