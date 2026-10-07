import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.sparse import diags_array

from poisson import Poisson

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float, sparse_bool=True):
        # Instantiate the implemented 1D class to borrow domain length; L
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case
        self.sparse_bool = sparse_bool

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N) # 1D vector denoted by N points
        
        # 2D Cartesian grid with ij indexing; mesh ordering --> x-coord vary along rows (down the cols) and y-coord along columns (up the rows)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=self.sparse_bool) #True boolean to store only leading 
        print(f"\n--- [MESH GENERATION] ---")
        print(f"Generated 2D grid for N={N} ({(N+1)}x{(N+1)} nodes).")
        if self.sparse_bool==True and N <= 6:
            print("X-coordinate matrix:\n", xij)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        print(f"\n--- [LAPLACE OPERATOR] ---")
        # 1. Calcuate spatial step assuming 0,1 domain; $delta_{x)=delta{y}$
        dx = self.p.L / N

        # 2. 2D Derivative matrix; D(2)
        # Boundary stencils left out; implemented separately
        D = diags_array([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(N+1, N+1))

        # 3. Scale by dx^2
        D_sc = D * (1.0 / dx**2)
        if N <= 6:
            print(f"1D Scaled Derivative Matrix (D_sc) shape {(N+1)}x{(N+1)}:\n", D_sc.toarray().round(2))
        # 4. Construct 1D Identity Matrix
        I = sparse.eye(N+1, format='lil')

        # 5. Assemble; A = (D_x ⊗ I_y) + (I_x ⊗ D_y)
        # block matrix calcuates Laplace on flattened 1D memory
        A = sparse.kron(D_sc, I) + sparse.kron(I, D_sc)

        return A.tolil()

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector

        Note
        ----
        Compute the Kronecker product of the 1D Laplace operator with itself
        to create the 2D Laplace operator. Then, assemble the right-hand side
        vector b by evaluating the function f at the mesh points and applying
        Dirichlet boundary conditions using the exact solution ue.

        """
        
        # 1. Create mesh and global coefficient matrix
        xij, yij = self.create_mesh(N)
        # Global coeff. matrix A
        A = self.laplace(N) #Compute the Kronecker product of the 1D Laplace operator with itsel to create the 2D Laplace operator

        print(f"\n--- [GLOBAL SYSTEM ASSEMBLY] ---")
        # 3. Evaluate the symbolic forcing func to create rhs vector
        F = self.meshfunction(f, xij, yij)

        # Flatten F into b vector
        b= F.ravel()
        print("Forcing function f(x,y) evaluated and flattened into vector 'b'.")
        # 3. Evaluate the exact solution to get the Dirichlet boundary values
        Ue = self.meshfunction(ue, xij, yij)
        b_boundary_values = Ue.ravel()

        # 4. Fetch the boundary indices
        bnds = self.get_boundary_indices(N)
        print(f"Identified {len(bnds)} boundary nodes out of {(N+1)**2} total nodes.")
        # 5. Overwrite the boundary nodes in A and b using fetched boundary indices
        for i in bnds:
            A[i, :] = 0.0          # Zero out the entire row
            A[i, i] = 1.0          # Place 1 on the main diagonal, forcing the eqn to be 1 dot u_i = ue_i
            b[i] = b_boundary_values[i] # Enforce the exact boundary value
        
        print("Dirichlet boundary conditions successfully enforced on A and b.")
        return A.tocsr(), b

    # ------- UTILS-------

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """
        x, y = sp.symbols('x y')
        # Convert mathematical expressions into numerical NumpY arrays
        f_func = sp.lambdify((x, y), u, modules='numpy') # https://docs.sympy.org/latest/modules/utilities/lambdify.html
        # Evaluate over mesh nodes
        U = f_func(xij,yij)

        if np.isscalar(U):
            U = np.full(xij.shape, U)
            
        return U

    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""

        B=np.ones((N+1,N+1), dtype=bool)
        B[1:-1, 1:-1]=0
        # Map the exact 1D endices that represent domain perimeter
        boundary_nodes = np.where(B.ravel() == 1)[0]

        return boundary_nodes

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error

        """
        N = u.shape[0] - 1
        dx = self.p.L / N
        dy = self.p.L / N
        
        #Recreate the mesh so we can evaluate the exact solution
        xij, yij = self.create_mesh(N)
        
        # Evaluate the exact SymPy solution ue onto the numerical grid
        ue_numerical = self.meshfunction(ue, xij, yij)
        
        # Calculate the squared difference between the numerical and exact solutions
        error_squared = (u - ue_numerical)**2
        
        # Sum the squared errors, multiply by the area element (dx*dy), and take the square root
        # scaled SSE evaluated at the nodes
        l2_err = np.sqrt(np.sum(error_squared) * dx * dy)
        
        return float(l2_err)

    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        print(f"\n======================================")
        print(f" INITIALIZING SOLVER FOR N={N}")
        print(f"======================================")

        A, b = self.assemble(N, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        print(f"\n--- [SOLVING] ---")
        print("Executing sparse linear solve (A \\ b)...")
        u = sparse_linalg.spsolve(A, b.ravel()).reshape((N + 1, N + 1))
        print("Solve complete. System reshaped back to 2D grid.")
        return u

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        from scipy.interpolate import interpn
        import numpy as np

        
        N = U.shape[0] - 1 #Determine N from the shape of the numerical solution matrix
        
        # Reconstruct the 1D coordinate axes used to build the grid
        xi = np.linspace(0, self.p.L, N + 1)
        yi = np.linspace(0, self.p.L, N + 1)
        
        # Apply the interpolation tool; method='linear' applies 1st-order 2D Lagrange basis functions.
        val = interpn((xi, yi), U, np.array([x, y]), method='linear')
        
        # interpn returns an array of results, so we extract the single float
        return float(val[0])



def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h, y: 1 - h / 2}).n()) < 1e-3


if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
