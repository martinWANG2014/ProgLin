# LinProg

Linear Programming Solver and Sensitivity Analysis Toolkit.

## Installation

```bash
pip install linprog
```

## Example
    # ============================================================
    # Example
    # max 3x1 + 5x2
    #
    # s.t.
    # x1          <= 4
    #      2x2    >= 8
    # 3x1 + 2x2   <= 18
    # x1 + x2     == 7
    # x1, x2 >= 0
    # ============================================================

```python
from linprog import LinearProgram

# name of decision variables
var_names = ["x1", "x2"]

# objective sense
objective_sense = "max" # "max" or "min"

# coefficients of the objective function
c = [3, 5]

# name of constraints
con_names = ["cte1", "cte3", "cte3", "cte4"]

# Matrix: each line represents the coefficients of a linear expression
# in the left part of a constraint
A = [
    [1, 0],
    [0, 2],
    [3, 2],
    [1, 1]
]

# sense of each constraint: "<=", ">=", or "=="
senses = ["<=", ">=", "<=", "=="]

# right hand side of each constraint
b = [4, 8, 18, 7]

# bounds on each variable: (min, max) or (min, None) or (None, max) or (None, None)
bounds = [(0, 7), (0, None)]

# type of each variable: 0 for continuous, 1 for integer
# A binary variable is handled as an integer variable bounded by 0 and 1.
var_types = [0]*2

lp = LinearProgram(
    c=c,
    A=A,
    senses=senses,
    b=b,
    objective=objective_sense,
    bounds=bounds,
    var_names=var_names,
    con_names=con_names,
    var_types=var_types
)

# solve the linear program
lp.solve()
# display the linear program model
print(lp.reportModel())
# display the solution
print(lp.solution())
# display the linear program in standard form
print(lp.reportStandardModelFormat())
# display the linear program in matrix form
print(lp.reportMatrixFormat())
# display the solution
print(lp.reportSolution())
# display the sensitivity analysis
print(lp.report_sensitive_analysis())
```