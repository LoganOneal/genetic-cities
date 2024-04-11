import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def solve_optimal_layout(chromosome: list[int], 
                         N: int, 
                         W: float, 
                         L: float, 
                         min_areas: list[float], 
                         min_widths: list[float],
                         max_area_weight: float, 
                         must_be_close: list[tuple[int, int]],
                         return_solution: bool=False) -> float:
    """
    Basis for the fitness function for a specific chromosome.
  
    Scores a chromosome defining N(N - 1)/2 relative positions of the N rooms in the facility
    using CVXPY to find the optimal layout.
  
    Parameters:
    chromosome: list of integers {0, 1, 2, 3} of length N(N - 1)/2, defining the relationship between room i and j
                as "i left of j", "j left of i", "i below j", or "j below i", respectively
    N: the number of rooms in the facility
    W: the width of the building (horizontal)
    L: the length of the building (vertical)
    min_areas: list of floats of length N, defining the minimum required area of each room
    max_area_weight: a weight defining the importance of maximizing the areas of the rooms with respect to
                     the other objectives (minimizing distance between certain areas)
    must_be_close: TODO

  
    Returns:
    float: the score of the optimal layout under the given constraints (higher score = better)
    """
    x, y, w, l = cp.Variable(N), cp.Variable(N), cp.Variable(N), cp.Variable(N)

    objective_func = max_area_weight * (-cp.sum(cp.log(w)) - cp.sum(cp.log(l))) # Objective: Maximize Areas
    objective_func += cp.sum([cp.norm1(cp.vstack([x[i] + (w[i]/2) - x[j] - (w[j]/2), y[i] + (l[i]/2) - y[j] - (l[j]/2)])) for (i, j) in must_be_close])
    constraints = [x >= 0, y >= 0, w >= 5, l >= 5, x + w <= W, y + l <= L]      # Boundary Constraints

    constraints += [cp.log(w) + cp.log(l) >= np.log(min_areas)]                 # Minimum Area Constraints
    # constraints += [w[i] >= min_widths[i] for i in np.where(~np.isnan(min_widths))[0]]  # Minimum Widths     
    # constraints += [l[i] >= min_widths[i] for i in np.where(~np.isnan(min_widths))[0]]
    constraints += [w - 5*l <= 0, l - 5*w <= 0]  # Maximum & Minimum Ratio Constraints: 1/5 <= w/l <= 5

    k = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Relative Positioning Constraints
            if chromosome[k] == 0:
                constraints += [x[i] + w[i] <= x[j]]
            elif chromosome[k] == 1:
                constraints += [x[j] + w[j] <= x[i]]
            elif chromosome[k] == 2:
                constraints += [y[i] + l[i] <= y[j]]
            else:  # chromosome[k] == 3
                constraints += [y[j] + l[j] <= y[i]]
            k += 1
    
    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except cp.error.SolverError:
        print("Error")
        return -np.inf

    if problem.status == "infeasible":
        return -np.inf
    if return_solution:
        return x.value, y.value, w.value, l.value
    return -problem.value

def plot_solution(N, x, y, w, l, W, L):
    fig, ax = plt.subplots()
    plt.scatter(x, y, s=0.1, color="black")
    ax.add_patch(Rectangle((0, 0), W, L, edgecolor="black", linestyle="dashed", fill=False))
    for i in range(N):
        ax.add_patch(Rectangle((x[i], y[i]), w[i], l[i], edgecolor="tab:blue", fill=False))
        plt.text(x[i] + (w[i] / 2) - 1, y[i] + (l[i] / 2) - 1, s=str(i))
    # plt.savefig("./images/solution.pdf")
    plt.show()
    plt.close()