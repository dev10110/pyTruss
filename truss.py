import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib import cm
import matplotlib as mpl
import pandas as pd
import scipy as sp
from scipy.optimize import NonlinearConstraint, Bounds
import warnings

cmap = cm.get_cmap('Blues', 100)


class Node:
    """Class to define a node.

    Args:
        x (float): x location of the node (meters)
        y (float): Description of parameter `y`.
        freex (bool): True if the node is allowed to deform in the x direction. Defaults to True.
        freey (bool): True if the node is allowed to deform in the y direction. Defaults to True.

    Examples
        See example usage code.

    Attributes:
        dx (float): Deflection of this node in the x direction, due to the applied loads (meters)
        dy (float): Deflection of this node in the y direction, due to the applied loads (meters)
        fx (float): Total force applied in the x direction (Newtons).
        fy (float): Total force applied in the y direction (Newtons).
        x
        y
        freex
        freey

    """
    def __init__(self, x, y, freex=True, freey=True):

        self.x = x
        self.y = y

        self.freex = freex
        self.freey = freey

        self.dx = 0
        self.dy = 0

        self.fx = 0
        self.fy = 0

    def pos(self):
        """Return node location.

        Returns:
            np.array: (x,y) location (meters)

        """

        return np.array([self.x, self.y])

    def __eq__(self, other):
        """Check if the node is the same as some other node.
        Only checks for position, not for whether the boundarys are defined the same"""

        return np.allclose(self.pos(), other.pos())

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f'N({self.x:2.2f}, {self.y:2.2f})'

    def plot(self):
        """Plots the node on the current axes.

        Color scheme:
            Black: Free in x,  Free  in y
            Blue:  Free in x,  Fixed in y
            Green: Fixed in x, Free  in y
            Red:   Fixed in x, Fixed in y

        Returns:
            None

        """
        if self.freex and self.freey:
            plt.plot(self.x, self.y, 'ko')
        elif self.freex and not self.freey:
            plt.plot(self.x, self.y, 'bo')
        elif not self.freex and self.freey:
            plt.plot(self.x, self.y, 'go')
        else:
            plt.plot(self.x, self.y, 'ro')

    def apply_load(self, fx, fy):
        """Sets the load at the node to (Fx, Fy).

        Args:
            fx (float): Load in x direction
            fy (float): Load in y direction

        Returns:
            None
        """

        self.fx = fx
        self.fy = fy

    def add_load(self, fx, fy):
        """Adds (fx, fy) load to the node. (Does not overwrite current values).

        Args:
            fx (float): Load in x direction (Newtons)
            fy (float): Load in y direction (Newtons)

        Returns:
            None
        """

        self.fx += fx
        self.fy += fy






class Bar:
    """Class to define bars in the truss. Currently only supports rectangular bar cross sections.

    Args:
        node0 (Node): Node at first end
        node1 (Node): Node at second end (order doesn't matter)
        w (float): Bar width (meters). Defaults to 5e-3.
        t (float): Bar thickness (meters). Defaults to 6.35e-3.
        E (float): Material Young's Modulus (Pa). Defaults to 71e9.
        yield_strength (float): Material Yield Strength (Pa). Defaults to 300e6.
        density (float): Material Density (kg/m3). Defaults to 2700.

    Attributes:
        node0
        node1
        w
        t
        E
        yield_strength
        density

    """

    def __init__(self, node0, node1, w=5e-3, t=6.35e-3, E=71e9, yield_strength=300e6, density=2700):

        if node0 == node1:
            raise ValueError(
                "Node 0 and Node 1 cannot be the same. Check if they are at the same place.")

        self.node0 = node0
        self.node1 = node1
        self.w = w
        self.t = t
        self.E = E
        self.yield_strength = yield_strength
        self.density = density

    def __eq__(self, other):
        # only checks if the same nodes are used, not for material or thicknesses.

        if (self.node0 == other.node0 and self.node1 == other.node1) or (self.node0 == other.node1 and self.node1 == other.node0):
            return True
        else:
            return False

    def __hash__(self):
        n0 = self.node0
        n1 = self.node1

        if n0.x < n1.x:
            return hash((n0, n1))
        if n0.x > n1.x:
            return hash((n1, n0))
        if n0.y < n1.y:
            return hash((n0, n1))
        if n0.y > n1.y:
            return hash((n1, n0))

        raise RuntimeError("Check if the two nodes of this bar are at the same place.")

    def __repr__(self):
        return f'B({self.node0}, {self.node1})'

    def length(self):
        """Returns bar length (meters)

        Returns:
            float: Bar length (meters)
        """

        v = self.node1.pos() - self.node0.pos()

        return np.sqrt(np.dot(v, v))


    def mid_point(self):
        """Returns coordinates of bar midpoint

        Returns:
            np.array: Mid point coordinations (meters)
        """
        return 0.5*(self.node0.pos()+self.node1.pos())

    def theta(self):
        """Get bar angle relative to the x axis

        Returns:
            float: Angle (radians)
        """

        v = self.node1.pos() - self.node0.pos()

        return np.arctan2(v[1], v[0])

    def e(self):
        """Unit direction vector of bar.

        Returns:
            np.array: (cos(theta), sin(theta))
        """
        th = self.theta()
        return np.array([np.cos(th), np.sin(th)])

    def area(self):
        """Returns bar cross sectional area (meters^2)

        Returns:
            float: bar cross sectional area (meters^2)
        """
        return self.w * self.t

    def volume(self):
        """Returns bar volume (meters^3)

        Returns:
            float: bar volume (meters^3)
        """

        return self.area()*self.length()

    def mass(self):
        """Returns bar mass (kg)

        Returns:
            float: bar mass (kg)
        """

        return self.volume()*self.density

    def I(self):
        """Returns bar second moment of area (meters^4)

        Returns:
            type: bar second moment of area (meters^4)
        """

        b = max(self.w, self.t)
        h = min(self.w, self.t)
        # can buckle in either direction, so need to compute both Is and take the smaller None
        I = b*h**3/12

        return I

    def EA(self):
        """Returns product of bar youngs modulus and cross sectional area.

        Returns:
            float: bar youngs modulus and cross sectional area. (Newtons)
        """
        return self.E * self.area()

    def stiffness(self):
        """Returns bar stiffness matrix, if both nodes were free.

        Returns:
            np.array: Bar stiffness matrix (SI units)
        """

        eeT = np.outer(self.e(), self.e())

        return (self.EA()/self.length())*np.block([[eeT, -eeT], [-eeT, eeT]])

    def extension(self):
        """Returns bar extension (meters), based on the last call to 'solve'.

        Returns:
            float: bar extension (meters)
        """

        du1 = np.array([self.node0.dx, self.node0.dy])
        du2 = np.array([self.node1.dx, self.node1.dy])

        return np.dot((du2-du1), self.e())

    def strain(self):
        """Returns bar strain (non-dimensional), based on the last call to 'solve'.

        Returns:
            float: bar strain (non-dimensional)
        """

        delta = self.extension()

        return delta/self.length()

    def stress(self):
        """Returns bar stress (Pa), based on the last call to 'solve'.

        Returns:
            float: bar stress (Pa)
        """

        strain = self.strain()

        return self.E * strain

    def tension(self):
        """Returns bar tension (N), based on the last call to 'solve'.

        Returns:
            float: bar tension (N)
        """

        return self.stress() * self.area()

    def buckling_load(self, K=1.0):
        """Compute the buckling load of a bar.
        Currently uses Eulers critical buckling load, and doesnt consider low slenderness beams.

        Args:
            K (float): Column effective length factor. Defaults to 1.0.

        Returns:
            float: Critical compressive load (Newtons)

        """

        Fcrit = np.pi**2*self.E*self.I()/(K*self.length())**2

        return Fcrit

    def qBuckle(self, K=1.0):
        """Returns whether the bar is buckled.

        Args:
            K (float): Column effective length factor. Defaults to 1.0.

        Returns:
            bool: True, if the bar is expected to buckle. Does not account for a safety factor.
        """

        if self.tension() >= 0.0:
            return False

        if abs(self.tension()) > self.buckling_load(K=K):
            return True
        else:
            return False

    def qYield(self):
        """Returns whether the bar stress has reached the yield strength.

        Args:
            K (float): Column effective length factor. Defaults to 1.0.

        Returns:
            bool: True, if the bar is expected to yield. Does not account for a safety factor.
        """
        if self.stress() <= 0.0:
            return False
        if self.stress() > self.yield_strength:
            return True
        else:
            return False


    def plot(self, color='k', def_scale=1.0, label=False):
        """Function to plot the bar on the current axes.

        Args:
            color: Color of deformed bar. Any type that pyplot can accept. Defaults to 'k'.
            def_scale (float): Scales deformations by `def_scale`. Defaults to 1.0.
            label (bool): If true, the bar widths are labelled. Defaults to False.

        Returns:
            None

        """
        nodes = [self.node0, self.node1]

        x = [n.x for n in nodes]
        y = [n.y for n in nodes]

        xdx = [n.x + def_scale*n.dx for n in nodes]
        ydy = [n.y + def_scale*n.dy for n in nodes]

        plt.plot(x, y, '0.8')
        plt.plot(xdx, ydy, color=color)

        if label:
            plt.text(*self.mid_point(), f'{1000*self.w:.2f} mm', horizontalalignment='center', verticalalignment='center')



class Truss:
    """Class to create a truss.

    Automatically removes repeated bars, but allows overlapping bars.

    Args:
        bars (list or tuple): List of bars that form the truss.
    """

    def __init__(self, bars):
        """Class to define a truss"""

        # remove duplicated bars
        self.bars = list(set(bars))

        # extract nodes
        self.nodes = self.extract_nodes(bars)

    @classmethod
    def from_delaunay(cls, nodes):
        """Class method to create a truss from a set of nodes, using a delaunay triangulation.

        Args:
            nodes (list or tuple): list of nodes to create the truss from.

        Returns:
            Truss: truss object

        Examples
            See the example usage scripts.
        """

        def find_neighbors(x, triang): return list(
            set(indx for simplex in triang.simplices if x in simplex for indx in simplex if indx != x))

        # create numpy nodes array
        node_points = np.vstack([n.pos() for n in nodes])

        # perform the triangulation
        d = Delaunay(node_points)

        bars = []

        for i, node in enumerate(nodes):

            neighbors = find_neighbors(i, d)
            for n in neighbors:
                bars.extend([Bar(node, nodes[n]) for n in neighbors])

        # note, there will be repeated bars here, but the clean up when creating the Truss will fix this issue.
        return cls(bars)

    @classmethod
    def from_fully_connected(cls, nodes):
        """Class method to create a truss by fully connecting every node with every other node.

        Args:
            nodes (listor tuple): List of nodes to create a truss from

        Returns:
            Truss: Truss using these nodes.
        """

        bars = set()

        for node0 in nodes:
            for node1 in nodes:
                if not node0 == node1:
                    bars.add(Bar(node0, node1))

        return cls(bars)

    def extract_nodes(self, bars):
        """Method to extract a unique list of nodes in the truss based on the set of bars.

        Args:
            bars (list): List of bars.

        Returns:
            list: list of unique nodes in the truss.

        """

        nodes = set()

        for bar in bars:
            nodes.add(bar.node0)
            nodes.add(bar.node1)

        return list(nodes)

    def set_widths(self, widths):
        """Helper function to set the widths of the bars in the truss to the passed vector of widths.
        Note the order is as will be in truss.bars.

        Args:
            widths (list or np.array): array of widths to set the widths of the bars to. (meters)

        Returns:
            None
        """

        for i, bar in enumerate(self.bars):
            bar.w = widths[i]

    def set_all_widths(self, width):
        """Sets the width of all elements in the truss to some width.

        Args:
            width (float): width (meters)
        """
        for b in self.bars:
            b.w = width

    def mass(self):
        """Returns the mass of the truss

        Returns:
            float: mass of the truss (kilogram)
        """

        return sum(bar.mass() for bar in self.bars)

    def solve(self, method="solve"):
        """Method to solve the truss.

        Args:
            method ("solve" or "lstsq"): method to solve the truss. 'solve' uses np.linalg.solve, while 'lstsq' uses np.linalg.lstsq, the least squares method. Defaults to "solve".

        Returns:
            None: Solution is set as a truss attribute, and the node deflections are stored into the nodes directly.

        Examples
            See example usage script.
        """
        # create stiffness matrix (we will delete the non-free nodes at the end)
        # for each node, assemble the stiffness matrix
        # 2*nodes because each node can have 2 degrees of freedom
        nodes = self.nodes
        bars = self.bars

        stiffness = np.zeros([2*len(nodes), 2*len(nodes)])

        for bar in self.bars:

            n0ind = nodes.index(bar.node0)
            n1ind = nodes.index(bar.node1)

            # replace all 16 elements
            bstiff = bar.stiffness()
            stiffness[2*n0ind:2*n0ind+2, 2*n0ind:2*n0ind+2] += bstiff[0:2, 0:2]
            stiffness[2*n0ind:2*n0ind+2, 2*n1ind:2*n1ind+2] += bstiff[0:2, 2:4]
            stiffness[2*n1ind:2*n1ind+2, 2*n0ind:2*n0ind+2] += bstiff[2:4, 0:2]
            stiffness[2*n1ind:2*n1ind+2, 2*n1ind:2*n1ind+2] += bstiff[2:4, 2:4]

        # create force vector
        F = np.zeros(2*len(self.nodes))
        for i, node in enumerate(self.nodes):
            F[2*i] = node.fx
            F[2*i+1] = node.fy

        # list the rows that are with fixed constraints
        delrows = []
        for i, node in enumerate(self.nodes):
            if node.freex == False:
                delrows.append(2*i)
            if node.freey == False:
                delrows.append(2*i+1)

        # delete rows and cols from the stiffness matrix
        stiffness = np.delete(stiffness, delrows, axis=0)
        stiffness = np.delete(stiffness, delrows, axis=1)

        # store into the stiffness of the truss
        self.stiffness = stiffness

        # delete rows from the force vector
        F = np.delete(F, delrows)
        # store into a force vector
        self.F = F

        # try to solve stiffness*deflections = forces for deflections
        try:
            if method == "solve":
                u = 10**-3 * np.linalg.solve(stiffness/10**6, F/10**3)

            elif method == "lstsq":
                sol = np.linalg.lstsq(stiffness, F)
                self.lstsq_sol = sol
                u = sol[0]

        except Exception as e:
            raise RuntimeError(f"Oops. Solve failed. \n {e}")

        self.u = u

        # reconstruct deformations
        ind = 0
        for i, node in enumerate(self.nodes):
            if node.freex:
                node.dx = u[ind]
                ind += 1
            # else skip the node y deflection
            if node.freey:
                node.dy = u[ind]
                ind += 1
            # else skip the node y deflection

        # does not return anything
        return None

    def plot(self,def_scale=1.0, figsize=(12, 5)):

        plt.figure(figsize=figsize)

        plt.subplot(121)

        self.plot_tensions(def_scale=def_scale)

        plt.subplot(122)

        self.plot_stress(def_scale=def_scale)


    def plot_tensions(self, ax=None, def_scale=1.0):

        if ax is None:
            ax = plt.gca()

        # create a symmetric stress bar
        Trange = max(abs(b.tension()) for b in self.bars)

        # plot all bars
        for bar in self.bars:

            if Trange < 0.1:
                c = '0.8'
            else:
                c = cmap((bar.tension()+Trange)/(2*Trange))

            bar.plot(color=c,def_scale=def_scale)
        for node in self.nodes:
            node.plot()

        self.plot_force_quiver(ax)

        # finally put in the colorbar:
        (cax, kw) = mpl.colorbar.make_axes(ax)
        norm = mpl.colors.Normalize(vmin=-Trange, vmax=+Trange)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_label('Bar Tensions (N)')

        return ax, cax

    def plot_force_quiver(self, ax):

        # plot the arrows
        x = [n.x for n in self.nodes if not (n.fx == 0 and n.fy == 0)]
        y = [n.y for n in self.nodes if not (n.fx == 0 and n.fy == 0)]
        fx = [n.fx for n in self.nodes if not (n.fx == 0 and n.fy == 0)]
        fy = [n.fy for n in self.nodes if not (n.fx == 0 and n.fy == 0)]

        ax.quiver(x, y, fx, fy, color='red',zorder=10)

    def plot_stress(self, ax=None, def_scale=1.0):

        if ax is None:
            ax = plt.gca()

        # create a symmetric stress bar
        Srange = max(abs(b.stress()) for b in self.bars)

        # plot all bars
        for bar in self.bars:

            if Srange < 0.1:
                c = '0.8'
            else:
                c = cmap((bar.stress()+Srange)/(2*Srange))

            bar.plot(color=c,def_scale=def_scale)
        for node in self.nodes:
            node.plot()

        # plot the arrows
        self.plot_force_quiver(ax)


        # finally put in the colorbar:
        (cax, kw) = mpl.colorbar.make_axes(ax)
        norm = mpl.colors.Normalize(vmin=-Srange, vmax=+Srange)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_label('Bar Stress (Pa)')

        return ax, cax



    def plot_widths(self, ax=None, def_scale=1.0, label=False):

        if ax is None:
            ax = plt.gca()

        # create a symmetric stress bar
        Wrange = 1000*max(b.w for b in self.bars)

        # plot all bars
        for bar in self.bars:

            if Wrange < 0.01e-3:
                c = '0.95'
            else:
                c = cmap(bar.w*1000/Wrange)

            bar.plot(color=c,def_scale=def_scale, label=label)

        for node in self.nodes:
            node.plot()

        # plot the arrows
        self.plot_force_quiver(ax)


        # finally put in the colorbar:
        (cax, kw) = mpl.colorbar.make_axes(ax)
        norm = mpl.colors.Normalize(vmin=0, vmax=Wrange)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_label('Bar Width (mm)')

        return ax, cax

    def details(self):
        """Returns detials on the nodes and the bars of the truss.

        Returns:
            (DataFrame, DataFrame): Tuple with pandas.dataframe of node details and bar details.
        """
        node_head = ["ID", "x (m)", "y (m)", "Free x?", "Free y?", "Force x (N)", "Force y (N)", "Delta x (mm)", "Delta y (mm)"]

        node_deets = [[i, n.x, n.y, n.freex, n.freey, n.fx, n.fy, n.dx*1000, n.dy*1000] for i, n in enumerate(self.nodes)]

        bar_head = ["ID", "Node 0","Node 1", "E (GPa)",  "Yield (MPa)", "w (mm)", "t (mm)", "A (mm2)", "I (mm4)", "L (m)", "m (kg)", "Buckling Load (N)", "T (N)", "ext (mm)", "Stress (MPa)", "Strain", "Will buckle?", "Will yield?", "Buckle Margin", "Yield Margin"]

        bar_deets = [[i, b.node0, b.node1, b.E/10**9, b.yield_strength/10**6, b.w*1000, b.t*1000, b.area()*10**6, b.I()*10**12, b.length(), b.mass(), b.buckling_load(), b.tension(), b.extension()*1000, b.stress()/10**6, b.strain(), b.qBuckle(), b.qYield(), -min(b.buckling_load()/b.tension(), 0), b.yield_strength/b.stress()] for i, b in enumerate(self.bars)]

        df_nodes = pd.DataFrame(node_deets, columns=node_head)

        df_bars = pd.DataFrame(bar_deets, columns=bar_head)

        return df_nodes, df_bars



    def tensions(self):
        """Returns list of bar tensions.

        Returns:
            list: list of bar tensions (N)
        """
        return [bar.tension() for bar in self.bars]

    def extensions(self):
        """Returns list of bar extensions

        Returns:
            list: list of bar extensions (meters)

        """
        return [bar.extension() for bar in self.bars]


    def minimize_mass(self, deflection_constraints=None, extra_constraints=None, buckling_SF=1.5, yield_SF=1.5, keep_feasible=False, method='SLSQP', minW=0.1e-3, maxW=20e-3, solve_method='lstsq', **kwargs):
        """Function to minimize the bar mass.

        Args:
            deflection_constraints (list): list of deflection constraints. Each member contains (Node, dx_min, dx_max, dy_min, dy_max) which specifies bounds. Use None if deflections dont matter. Defaults to None.
            extra_constraints (list): list of extra NonlinearConstraint objects to pass to the optimizer. Defaults to None.
            buckling_SF (float): Safety factor on buckling constraints. Set to 0 if buckling doesnt matter.  Defaults to 1.5.
            yield_SF (float): Safety factor on yield constraints. Set to 0 if yield constraints dont matter. Defaults to 1.5.
            keep_feasible (bool): True if all tested widths have to be feasible during the optmization. Defaults to False.
            method (str): method used for the optimization. Defaults to 'SLSQP'.
            minW (float): minimum width of bars (meters). Defaults to 0.1e-3.
            maxW (float): maximum width of bars (meters). Defaults to 20e-3.
            solve_method (str): method to pass to truss.solve().  Defaults to 'lstsq'.

        Returns:
            np.optimize.result: Optimization result. Also applies the solution to the structure and does one final solve.

        """

        def f_objective(widths):

            self.set_widths(widths)

            return self.mass()


        def f_deflection_con(widths):

            self.set_widths(widths)

            self.solve(method=solve_method)

            con = []

            for defl in deflection_constraints:
                node, dx_min, dx_max, dy_min, dy_max = defl

                if dx_min is not None:
                    con.append(node.dx - dx_min)
                if dx_max is not None:
                    con.append(dx_max - node.dx)
                if dy_min is not None:
                    con.append(node.dy - dy_min)
                if dy_max is not None:
                    con.append(dy_max - node.dy)

            return con

        def f_buckling_con(widths):
            self.set_widths(widths)

            self.solve(method=solve_method)

            return [buckling_SF*bar.tension() + bar.buckling_load() for bar in self.bars]

        def f_yield_con(widths):
            self.set_widths(widths)

            self.solve(method=solve_method)

            return [bar.yield_strength - yield_SF*bar.stress() for bar in self.bars]

        buckling_constraint = NonlinearConstraint(f_buckling_con, lb=0, ub=np.inf, keep_feasible=keep_feasible)
        yield_constraint = NonlinearConstraint(f_yield_con, lb=0, ub=np.inf, keep_feasible=keep_feasible)
        constraints = [buckling_constraint, yield_constraint]


        if deflection_constraints is not None:
            deflection_constraint = NonlinearConstraint(f_deflection_con, lb=0, ub=np.inf, keep_feasible=keep_feasible)
            constraints.append(deflection_constraint)

        if extra_constraints is not None:
            constraints.extend(extra_constraints)

        widths0 = [bar.w for bar in self.bars]

        bounds = Bounds(minW, maxW, keep_feasible=keep_feasible)

        sol = sp.optimize.minimize(f_objective, x0=widths0, bounds=bounds, constraints=constraints, method=method, **kwargs)

        if not sol.success:
            warnings.warn("Optimization hasn't been successful! Please check!")


        self.set_widths(sol.x)
        self.solve(method=solve_method)

        return sol

    def qBuckle(self, K=1.0):
        """Returns true if any element in the truss with buckle. Unlikely to be accurate.

        Args:
            K (float): Euler effective length parameter. Defaults to 1.0

        Returns:
            bool: True if any element with buckle.
        """

        for b in self.bars:
            if b.qBuckle(K=K):
                return True

        return False

    def qYield(self):
        """Returns true if any element in the truss with yield.

        Returns:
            bool: True if any element with yield.
        """
        for b in self.bars:
            if b.qYield():
                return True

        return False
