import numpy as np
import pandas as pd
from scipy.optimize import linprog
TOL = 1e-6
#####
## Contact: chenghao.wang@uphf.fr
#####
class LinearProgram:
    def __init__(
        self,
        c,
        A,
        senses,
        b,
        objective="max",
        bounds=None,
        var_names=None,
        con_names=None
    ):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.senses = senses
        self.b = np.array(b, dtype=float)
        self.objective = objective.lower()

        self.m, self.n = self.A.shape

        self.var_names = var_names or [f"x{j+1}" for j in range(self.n)]
        self.con_names = con_names or [f"Constraint {i+1}" for i in range(self.m)]
        self.bounds = bounds or [(0, None)] * self.n

        self.res = None
        self.objective_value = None

    # ------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------

    @staticmethod
    def fmt(v):
        if np.isneginf(v):
            return "-INFINITY"
        if np.isposinf(v):
            return "INFINITY"
        return round(float(v), 6)

    @staticmethod
    def linear_expr(coeffs, names):
        terms = []
        for a, name in zip(coeffs, names):
            if abs(a) < TOL:
                continue
            if abs(a - 1) < TOL:
                terms.append(name)
            elif abs(a + 1) < TOL:
                terms.append(f"-{name}")
            else:
                terms.append(f"{a:g} {name}")

        if not terms:
            return "0"

        return " + ".join(terms).replace("+ -", "- ")

    @staticmethod
    def interval_to_change(current, low, high):
        dec = np.inf if np.isneginf(low) else current - low
        inc = np.inf if np.isposinf(high) else high - current
        return dec, inc

    # ------------------------------------------------------------
    # Model reports
    # ------------------------------------------------------------

    def reportModel(self):
        lines = []
        lines.append("ORIGINAL LP MODEL")
        lines.append("=" * 80)

        obj = self.linear_expr(self.c, self.var_names)
        lines.append(f"{self.objective.upper()} z = {obj}")
        lines.append("")
        lines.append("Subject to:")

        for name, row, sense, rhs in zip(
            self.con_names, self.A, self.senses, self.b
        ):
            lhs = self.linear_expr(row, self.var_names)
            lines.append(f"  {name}: {lhs} {sense} {rhs:g}")

        lines.append("")
        lines.append("Bounds:")
        for name, bound in zip(self.var_names, self.bounds):
            lb, ub = bound
            if lb is not None:
                lines.append(f"  {name} >= {lb:g}")
            if ub is not None:
                lines.append(f"  {name} <= {ub:g}")

        lines.append("-" * 80)
        return "\n".join(lines)

    def _build_standard_form(self):
        rows = []
        rhs_std = []
        slack_names = []

        for i, (row, sense, rhs) in enumerate(
            zip(self.A, self.senses, self.b)
        ):
            if sense == "<=":
                rows.append(row)
                rhs_std.append(rhs)

            elif sense == ">=":
                rows.append(-row)
                rhs_std.append(-rhs)

            elif sense == "==":
                rows.append(row)
                rhs_std.append(rhs)

            else:
                raise ValueError("Each sense must be '<=', '>=', or '=='")

        A_base = np.array(rows, dtype=float)
        b_std = np.array(rhs_std, dtype=float)

        slack_cols = []

        for i, sense in enumerate(self.senses):
            if sense in ["<=", ">="]:
                col = np.zeros(self.m)
                col[i] = 1.0
                slack_cols.append(col)
                slack_names.append(f"s{i+1}")

        if slack_cols:
            S = np.column_stack(slack_cols)
            Astd = np.hstack([A_base, S])
        else:
            Astd = A_base.copy()

        cstd = np.concatenate([self.c, np.zeros(len(slack_names))])
        std_names = self.var_names + slack_names

        return Astd, b_std, cstd, std_names, slack_names

    def reportStandardModelFormat(self):
        Astd, b_std, cstd, std_names, _ = self._build_standard_form()

        lines = []
        lines.append("STANDARD LP FORMAT")
        lines.append("=" * 80)

        obj = self.linear_expr(cstd, std_names)
        lines.append(f"{self.objective.upper()} z = {obj}")
        lines.append("")
        lines.append("Subject to:")

        for row, rhs in zip(Astd, b_std):
            lhs = self.linear_expr(row, std_names)
            lines.append(f"  {lhs} = {rhs:g}")

        lines.append("")
        lines.append("All variables >= 0")
        lines.append("-" * 80)
        return "\n".join(lines)

    def reportMatrixFormat(self):

        Astd, b_std, cstd, std_names, _ = self._build_standard_form()

        def pretty_matrix(M):
            rows = []

            for i, row in enumerate(M):
                body = " ".join(f"{x:>6g}" for x in row)

                if i == 0:
                    rows.append(f"⎡ {body} ⎤")
                elif i == len(M) - 1:
                    rows.append(f"⎣ {body} ⎦")
                else:
                    rows.append(f"⎢ {body} ⎥")

            return rows

        def pretty_vector(v):
            rows = []

            for i, x in enumerate(v):

                if i == 0:
                    rows.append(f"⎡ {x:g} ⎤")
                elif i == len(v) - 1:
                    rows.append(f"⎣ {x:g} ⎦")
                else:
                    rows.append(f"⎢ {x:g} ⎥")

            return rows

        def pretty_variable_vector(names):
            rows = []

            for i, n in enumerate(names):

                if i == 0:
                    rows.append(f"⎡ {n} ⎤")
                elif i == len(names) - 1:
                    rows.append(f"⎣ {n} ⎦")
                else:
                    rows.append(f"⎢ {n} ⎥")

            return rows

        lines = []

        lines.append("MATRIX FORMAT")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"{self.objective.upper()} z = cᵀx")
        lines.append("")
        lines.append("Subject To")
        lines.append("")
        lines.append("Ax = b")
        lines.append("x ≥ 0")
        lines.append("")

        # A matrix
        A_lines = pretty_matrix(Astd)

        lines.append("A =")
        lines.extend(A_lines)
        lines.append("")

        # x vector
        x_lines = pretty_variable_vector(std_names)

        lines.append("x =")
        lines.extend(x_lines)
        lines.append("")

        # b vector
        b_lines = pretty_vector(b_std)

        lines.append("b =")
        lines.extend(b_lines)
        lines.append("")

        # c row vector
        c_str = " ".join(f"{v:g}" for v in cstd)

        lines.append(f"cᵀ = [ {c_str} ]")
        lines.append("-" * 80)

        return "\n".join(lines)

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------

    def solve(self):
        A_ub, b_ub = [], []
        A_eq, b_eq = [], []

        for row, sense, rhs in zip(self.A, self.senses, self.b):
            if sense == "<=":
                A_ub.append(row)
                b_ub.append(rhs)
            elif sense == ">=":
                A_ub.append(-row)
                b_ub.append(-rhs)
            elif sense == "==":
                A_eq.append(row)
                b_eq.append(rhs)
            else:
                raise ValueError("Each sense must be '<=', '>=', or '=='")

        scipy_c = self.c.copy()

        if self.objective == "max":
            scipy_c = -scipy_c
        elif self.objective != "min":
            raise ValueError("objective must be 'max' or 'min'")

        self.res = linprog(
            scipy_c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=self.bounds,
            method="highs-ds"
        )

        if not self.res.success:
            raise RuntimeError(self.res.message)

        self.objective_value = float(self.c @ self.res.x)

        return self.res

    def solution(self):
        if self.res is None:
            self.solve()

        x = np.asarray(self.res.x).copy()
        x[np.abs(x) <= 1e-6] = 0.0

        sol = {
            v: float(val)
            for v, val in zip(self.var_names, x)
        }
        sol["obj"] = self.objective_value
        return sol

    def reportSolution(self):
        if self.res is None:
            self.solve()

        x = self.res.x

        solution_table = pd.DataFrame({
            "Variable": self.var_names,
            "Value": x,
        })

        lines = []
        lines.append("OPTIMAL SOLUTION")
        lines.append("=" * 80)

        lines.append(f"Objective Type : {self.objective.upper()}")
        lines.append(
            f"Optimal Objective Value : {self.objective_value:.6f}"
        )
        lines.append("")

        lines.append(solution_table.to_string(index=False))
        lines.append("-" * 80)
        report = "\n".join(lines)
        return report
        # return {
        #     "objective_type": self.objective,
        #     "objective_value": self.objective_value,
        #     "solution_table": solution_table,
        #     "report_text": report
        # }

    # ------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------

    def report_sensitive_analysis(self):
        if self.res is None:
            self.solve()

        x = self.res.x
        true_obj = self.objective_value

        Astd, b_std, cstd, std_names, slack_names = self._build_standard_form()

        cstd_sens = -cstd if self.objective == "min" else cstd.copy()

        std_values = list(x)

        for i, sense in enumerate(self.senses):
            activity = self.A[i] @ x

            if sense == "<=":
                std_values.append(self.b[i] - activity)
            elif sense == ">=":
                std_values.append(activity - self.b[i])

        zstd = np.array(std_values, dtype=float)

        total_vars = Astd.shape[1]
        rank_needed = Astd.shape[0]

        basic_idx = list(np.where(zstd > TOL)[0])

        for j in range(total_vars):
            if len(basic_idx) == rank_needed:
                break
            if j not in basic_idx:
                trial = basic_idx + [j]
                if np.linalg.matrix_rank(Astd[:, trial]) == len(trial):
                    basic_idx.append(j)

        basic_idx = basic_idx[:rank_needed]
        nonbasic_idx = [j for j in range(total_vars) if j not in basic_idx]

        B = Astd[:, basic_idx]
        N = Astd[:, nonbasic_idx]

        if np.linalg.matrix_rank(B) < rank_needed:
            raise RuntimeError("Could not identify nonsingular basis.")

        B_inv = np.linalg.inv(B)
        xB = B_inv @ b_std

        cB = cstd_sens[basic_idx]
        cN = cstd_sens[nonbasic_idx]

        y = cB @ B_inv
        reduced = cstd_sens - y @ Astd

        # --------------------------------------------------------
        # Solution table
        # --------------------------------------------------------

        display_reduced = reduced[:self.n]

        # if self.objective == "max":
        display_reduced = -display_reduced

        display_reduced[np.abs(display_reduced) < TOL] = 0.0
        solution_table = pd.DataFrame({
            "Variable": self.var_names,
            "Value": x,
            "Reduced Cost": display_reduced
        })

        # --------------------------------------------------------
        # Constraint table
        # --------------------------------------------------------

        con_rows = []

        for i, (row, sense, rhs) in enumerate(
            zip(self.A, self.senses, self.b)
        ):
            activity = row @ x

            if sense == "<=":
                slack = rhs - activity
            elif sense == ">=":
                slack = activity - rhs
            else:
                slack = 0.0

            shadow = y[i]

            if sense == ">=":
                shadow = -shadow

            if abs(shadow) < TOL:
                shadow = 0.0
            con_rows.append({
                "Constraint": self.con_names[i],
                "Slack/Surplus": slack,
                "Shadow/Dual Price": shadow
            })

        constraint_table = pd.DataFrame(con_rows)

        # --------------------------------------------------------
        # Objective coefficient sensitivity
        # --------------------------------------------------------

        obj_rows = []
        YN = B_inv @ N
        reduced_N = cN - cB @ YN

        for j in range(self.n):
            current_coef_sens = cstd_sens[j]

            if j in nonbasic_idx:
                pos = nonbasic_idx.index(j)
                r = reduced_N[pos]

                lo_sens = -np.inf
                hi_sens = current_coef_sens - r

            else:
                k = basic_idx.index(j)
                row = YN[k, :]

                low_delta, high_delta = -np.inf, np.inf

                for r, a in zip(reduced_N, row):
                    if abs(a) < TOL:
                        continue

                    bound = r / a

                    if a > 0:
                        low_delta = max(low_delta, bound)
                    else:
                        high_delta = min(high_delta, bound)

                lo_sens = current_coef_sens + low_delta
                hi_sens = current_coef_sens + high_delta

            if self.objective == "min":
                lo = -hi_sens
                hi = -lo_sens
            else:
                lo = lo_sens
                hi = hi_sens

            dec, inc = self.interval_to_change(self.c[j], lo, hi)

            obj_rows.append({
                "Variable": self.var_names[j],
                "Current c_j": self.c[j],
                "Allowable Increase": self.fmt(inc),
                "Allowable Decrease": self.fmt(dec)
            })

        obj_sensitivity = pd.DataFrame(obj_rows)

        # --------------------------------------------------------
        # RHS sensitivity
        # --------------------------------------------------------

        rhs_rows = []

        for i in range(self.m):
            d = B_inv[:, i]
            low_delta, high_delta = -np.inf, np.inf

            for xb, di in zip(xB, d):
                if abs(di) < TOL:
                    continue

                bound = -xb / di

                if di > 0:
                    low_delta = max(low_delta, bound)
                else:
                    high_delta = min(high_delta, bound)

            std_lo = b_std[i] + low_delta
            std_hi = b_std[i] + high_delta

            if self.senses[i] == ">=":
                orig_lo = -std_hi
                orig_hi = -std_lo
            else:
                orig_lo = std_lo
                orig_hi = std_hi

            dec, inc = self.interval_to_change(
                self.b[i], orig_lo, orig_hi
            )

            rhs_rows.append({
                "Constraint": self.con_names[i],
                "Current RHS": self.b[i],
                "Allowable Increase": self.fmt(inc),
                "Allowable Decrease": self.fmt(dec)
            })

        rhs_sensitivity = pd.DataFrame(rhs_rows)



        basis_table = pd.DataFrame({
            "Basic Variable": [std_names[i] for i in basic_idx],
            "Value": xB
        })

        report_text = "\n\n".join([
            # self.reportModel(),
            # self.reportStandardModelFormat(),
            # self.reportMatrixFormat(),
            "OPTIMAL SOLUTION\n" + "=" * 80 + "\n"
            + f"Objective value = {true_obj:.6f}\n\n"
            + solution_table.to_string(index=False)+ "\n"+"-" * 80,
            "CONSTRAINT REPORT\n" + "=" * 80 + "\n"
            + constraint_table.to_string(index=False)+ "\n"+"-" * 80,
            "OBJECTIVE COEFFICIENT SENSITIVITY\n" + "=" * 80 + "\n"
            + obj_sensitivity.to_string(index=False)+ "\n"+"-" * 80,
            "RHS SENSITIVITY\n" + "=" * 80 + "\n"
            + rhs_sensitivity.to_string(index=False)+ "\n"+"-" * 80
            # ,
            # "FINAL BASIS\n" + "=" * 80 + "\n"
            # + basis_table.to_string(index=False)+ "\n"+"-" * 80
        ])
        return report_text

        # print(report_text)
        #
        # return {
        #     "result": self.res,
        #     "objective_value": true_obj,
        #
        #     "lp_format": self.reportModel(),
        #     "standard_lp_format": self.reportStandardModelFormat(),
        #     "matrix_format": self.reportMatrixFormat(),
        #
        #     "solution": solution_table,
        #     "constraints": constraint_table,
        #     "obj_sensitivity": obj_sensitivity,
        #     "rhs_sensitivity": rhs_sensitivity,
        #     "basis": basis_table,
        #
        #     "A_standard": Astd,
        #     "b_standard": b_std,
        #     "c_standard": cstd,
        #     "standard_variable_names": std_names,
        #
        #     "report_text": report_text
        # }

if __name__ == '__main__':
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

    c = [3, 5]

    A = [
        [1, 0],
        [0, 2],
        [3, 2],
        [1, 1]
    ]

    senses = ["<=", ">=", "<=", "=="]

    b = [4, 8, 18, 7]

    bounds = [(0, None), (0, None)]

    lp = LinearProgram(
        c=c,
        A=A,
        senses=senses,
        b=b,
        objective="max",
        bounds=bounds,
        var_names=["x1", "x2"],
        con_names=["cte1", "cte3", "cte3", "cte4"]
    )
    lp.solve()
    # display the linear program model
    print(lp.reportModel())
    # display the linear program in standard form
    print(lp.reportStandardModelFormat())
    # display the linear program in matrix form
    print(lp.reportMatrixFormat())
    # display the solution
    print(lp.reportSolution())
    # display the sensitivity analysis
    print(lp.report_sensitive_analysis())

    print(lp.solution())
