# Genetic Algorithm for the Traveling Salesman Problem (TSP)

A small Python project implementing a **Genetic Algorithm (GA)** to find a short route for the **Traveling Salesman Problem (TSP)** on a randomly generated set of cities.

The script generates random 2D city coordinates, initializes a population of candidate routes (permutations), and iteratively improves the best route using:
- **Selection** (elitism)
- **Ordered crossover (OX)**
- **Swap mutation**

It also visualizes:
- the random city layout,
- the current best route,
- and the improvement of the best fitness value over iterations.
