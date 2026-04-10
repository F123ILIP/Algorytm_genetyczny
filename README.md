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

## Demo / Example output
The repository includes example plots in the current README screenshots.

## Requirements
This project uses common scientific Python libraries:
- NumPy
- Pandas
- Matplotlib
- SciPy

Install them with:

```bash
pip install numpy pandas matplotlib scipy
```

## How to run
Run the main script:

```bash
python main.py
```

The program is interactive and will ask you for parameters like:
- number of cities
- population size
- crossover probability
- mutation probability

## Notes on the implementation
- **Fitness / objective**: minimize the total route length (including return to the starting city).
- **Distance matrix** is computed using Euclidean distance (`scipy.spatial.distance.cdist`).
- Routes are represented as permutations of indices.

## Reproducibility
For reproducible runs, consider adding a fixed random seed in `main.py` (NumPy + Python `random`).

## Roadmap
- [ ] Add a `requirements.txt`
- [ ] Add a non-interactive CLI mode (arguments instead of `input()`)
- [ ] Add configuration presets and better logging
- [ ] Add unit tests for crossover and mutation operators

## License
If you want, I can add an MIT license file.