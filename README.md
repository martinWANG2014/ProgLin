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

lp = LinearProgram(
    c=[3,5],
    A=[
        [1,0],
        [0,2],
        [3,2],
        [1,1]
    ],
    senses=["<=",">=","<=","=="],
    b=[4,8,18,7],
    objective="max",
    bounds=[(0, None), (0, None)],
    var_names=["x1", "x2"],
    con_names=["cte1", "cte3", "cte3", "cte4"],
)

lp.solve()
# display the linear program model
print(lp.reportModel())
# display the linear program in standard form
print(lp.reportStandardModelFormat())
# display the linear program in matrix form
print(lp.reportMatrixFormat())
# solution of the linear program
print(lp.solution)
# display the solution
print(lp.reportSolution())
# display the sensitivity analysis
print(lp.report_sensitive_analysis())
```
