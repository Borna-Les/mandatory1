import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import diags_array

x, y, t = sp.symbols("x,y,t")

class Wave2D():
    """Class for solving the 2D wave equation"""

    def __init__(self, L: float = 1, c: float=1.0):
        self.L = L
        self.c = c
        self.mx = 3  # Default wave param
        self.my = 3  # Default wave param

    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        
        xi =np.linspace(0,self.L, N+1)
                
        #2D Cartesian grid with ij indexing; mesh ordering --> x-coord vary along rows (down the cols) and y-coord along columns (up the rows)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=sparse) #True boolean to store only leading 
        print(f"\n--- [MESH GENERATION] ---")
        print(f"Generated 2D grid for N={N} ({(N+1)}x{(N+1)} nodes).")
        if sparse==True and N <= 6:
            print("X-coordinate matrix:\n", xij.round(2))

        return xij, yij
        

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        # 2D Derivative matrix; D(2)
        D = diags_array([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(N+1, N+1)) #isolate toeplitz structure; rows 0<j<N of D_2*u_n becomes unjp1 - 2unj + unmj
        return D.tolil()

    @property
    def w(self):
        """Return the dispersion coefficient"""
        import math

        kx = self.mx*math.pi /self.L
        ky = self.my*math.pi/self.L
        return self.c * math.sqrt(kx**2 + ky**2)

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def get_exact_mesh(self, t_eval: float) -> np.ndarray:
        """Helper to evaluate the exact standing wave at any given time t"""
        xij, yij = self.create_mesh(self.N, sparse=False)
        exact_expr = self.ue(self.mx, self.my)
        ue_func = sp.lambdify((x, y, t), exact_expr, modules='numpy') # Note: using global t
        return ue_func(xij, yij, t_eval) # Use the passed argument 't_eval

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.N=N
        self.mx=mx
        self.my=my

        # Call the helper for t = 0
        U_0 = self.get_exact_mesh(0.0)
        # Call the helper for t = -dt
        U_m1 = self.get_exact_mesh(-self.dt)

        return U_0, U_m1

    @property
    def dt(self) -> float:
        """Return the time step"""
        dx = self.L/self.N
        dt = self.cfl*dx/self.c
        return dt

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        N = u.shape[0] - 1
        dx = self.L / N
        dy = self.L / N
        
        # xij, yij = self.create_mesh(N)
        # ue = self.ue(self.mx, self.my)
        # # Evaluate the exact SymPy solution ue onto the numerical grid
        # ue_numerical = self.meshfunction(ue, xij, yij)
        ue_numerical = self.get_exact_mesh(t0)
        # Calculate the squared difference between the numerical and exact solutions
        error_squared = (u - ue_numerical)**2
        # Sum the squared errors, multiply by the area element (dx*dy), and take the square root
        # scaled SSE evaluated at the nodes
        l2_err = np.sqrt(np.sum(error_squared) * dx * dy)
        
        return float(l2_err)

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.N= N
        self.c = c
        self.mx=mx
        self.my=my

        h = self.L / self.N
        
        #Get the initial states from a separate func
        U_nm1, U_n = self.initialize(self.N, self.mx, self.my)
        # Prepare the unscaled 1D differentiation matrix
        D = self.D2(N).toarray()
        # The squared Courant number
        C2 = (self.c * self.dt / h)**2 
        
        # Dictionary to store data if requested
        data = {}
        if store_data > 0:
            data[0] = U_nm1.copy()
            data[1] = U_n.copy()

        #The Main Time Loop; Start at n=1 as we've already computed U0 and U1
        for n in range(1, Nt):
            # D @ U_n computes derivatives along one axis, U_n @ D.T computes along the other
            U_np1 = 2 * U_n - U_nm1 + C2 * (D @ U_n + U_n @ D.T)
            self.apply_bcs(U_np1)
            if store_data > 0 and (n + 1) % store_data == 0:
                data[n + 1] = U_np1.copy()
            # Shift states forward in time for the next iteration
            U_nm1 = U_n.copy()
            U_n = U_np1.copy()

        # Return logic based on the test harness requirements
        if store_data == -1:
            final_time = Nt * self.dt
            error = self.l2_error(U_n, final_time)
            return h, error
        else:
            return data

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)

def test_convergence_wave2d():
    sol = Wave2D()
    print("Running convergence tests... (This checks multiple grid sizes)")
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    print("\nConvergence Rates (r):", np.round(r, 4))
    print("Final Rate:", round(r[-1], 4))
    assert abs(r[-1] - 2) < 1e-2, f"Convergence rate {r[-1]} is not approaching 2.0"
    print("\nSUCCESS! The Wave2D solver is verified second-order accurate.")


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        D = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N+1, N+1)).tolil()
        # Apply Neumann "ghost node" idea to the matrix boundaries directly
        D[0, 0] = -2.0
        D[0, 1] = 2.0
        D[-1, -1] = -2.0
        D[-1, -2] = 2.0
        return D

    def ue(self, mx: int, my: int) ->sp.Expr:
        return sp.cos(mx*sp.pi * x/self.L) * sp.cos(my* sp.pi * y /self.L) *sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        pass # Implemented modifying D2 directly...

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    print("Running convergence tests for Neumann BC case... (This checks multiple grid sizes)")
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    print("\nConvergence Rates (r):", np.round(r, 4))
    print("Final Rate:", round(r[-1], 4))
    assert abs(r[-1] - 2) < 0.05, f"Convergence rate {r[-1]} is not approaching 2.0"
    print("\nSUCCESS! The Wave2D_Neumann solver is verified second-order accurate.")

def test_exact_wave2d():
    # Set stated mathematical conditions
    magic_cfl = 1.0 / np.sqrt(2)
    mx_val = 2
    my_val = 2  # mx must equal my
    
    # Adding some grid size and time steps...
    N = 20
    Nt = 10
    #Test the Dirichlet Problem (Wave2D)
    sol_D = Wave2D()
    h_D, err_D = sol_D(N=N, Nt=Nt, cfl=magic_cfl, mx=mx_val, my=my_val, store_data=-1)
    
    #Test the Neumann Problem (Wave2D_Neumann )
    sol_N = Wave2D_Neumann()
    h_N, err_N = sol_N(N=N, Nt=Nt, cfl=magic_cfl, mx=mx_val, my=my_val, store_data=-1)
    
    # Assert both errors are below the requested threshold (10^-12 )
    assert err_D < 1e-12, f"Dirichlet exact solution failed. Error: {err_D}"
    assert err_N < 1e-12, f"Neumann exact solution failed. Error: {err_N}"
    
    print(f"Exact test passed!")
    print(f"Dirichlet Error: {err_D:.2e}")
    print(f"Neumann Error:   {err_N:.2e}")

if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
