import numpy as np
import matplotlib.pyplot as plt

EPSILON_0 = 1.0
MU_0 = 1.0

def interleaved(ct, dt):
    if ct % dt == 0:
        return ct
    else:
        return ct - round(dt - (ct % dt), 5) 


class Mesh():
    def __init__(self, initial_position, final_position, dx, sub_dx = None, refined_initial_index = None, refined_final_index = None):
        self.dx = dx
        self.p_i = initial_position
        self.p_f = final_position
        self.sub_dx = sub_dx
        self.refined_initial_index = refined_initial_index
        self.refined_final_index = refined_final_index

        if sub_dx is None:
            self.xE = np.linspace(self.p_i, self.p_f, num = int(1 + (self.p_f-self.p_i)/dx))
        elif (refined_final_index is not None) and (refined_initial_index is not None):
            t1 = np.linspace(self.p_i, self.p_i + self.dx * (refined_initial_index-1), num = refined_initial_index)
            t3 = np.linspace(self.p_i + self.dx * (refined_final_index+1), self.p_f, num = int((self.p_f-self.p_i)/dx-refined_final_index))

            self.xE = np.hstack((t1, np.linspace(t1[-1]+dx, t3[0]-dx, num = int(1+(refined_final_index-refined_initial_index)*dx / sub_dx)),t3))
            self.refined_final_index = np.where(self.xE==(self.p_i + self.dx * (refined_final_index+1)))[0][0]
        else:
            raise ValueError("Invalid inputs. Please, give your decisions a second thought :/")
        
        self.xH = (self.xE[1:] + self.xE[:-1]) / 2.0

class FDTD1D():
    def __init__(self, mesh, boundary, CFL , relative_epsilon_vector=None, SolverType = None):
        self.mesh = mesh
        self.xE = self.mesh.xE
        self.xH = self.mesh.xH
        self.CFL = CFL
        self.SolverType = SolverType


        self.E = np.zeros(self.xE.shape)
        self.H = np.zeros(self.xH.shape)

        self.dx = self.mesh.dx
        self.sub_dx= self.mesh.sub_dx
        self.sources = []
        self.t = 0.0

        if relative_epsilon_vector is None:
            self.epsilon_r = np.ones(self.xE.shape)
        else:
            self.epsilon_r = relative_epsilon_vector
        self.boundary = boundary

    def addSource(self, source):
        self.sources.append(source)
     
    def setE(self, fieldE):
        self.E[:] = fieldE[:]

    def setH(self, fieldH):
        self.H[:] = fieldH[:]

    def getE(self):
        fieldE = np.zeros(self.E.shape)
        fieldE = self.E[:]
        return fieldE
    
    def getH(self):
        fieldH = np.zeros(self.H.shape)
        fieldH = self.H[:]
        return fieldH

    def stepDefault(self):
        self.dt = self.CFL * self.dx
        E = self.E
        H = self.H
        c = self.dt/self.dx
        c_eps = np.ones(self.epsilon_r.size)
        c_eps[:] = self.dt/self.dx / self.epsilon_r[:]
        E_aux_izq = E[1]
        E_aux_dch= E[-2]

        H += - self.dt/self.dx *(E[1:] - E[:-1])
        for source in self.sources:
            H[source.location] += source.function(self.t + self.dt/2)

        E[1:-1] += - c_eps[1:-1] * (H[1:] - H[:-1])
        for source in self.sources:
            E[source.location] += source.function(self.t)
        self.t += self.dt

        if self.boundary == "pec":
            E[0] = 0.0
            E[-1] = 0.0
        elif self.boundary == "pmc":
            E[0] = E[0] - c / self.epsilon_r[0] * (2 * H[0])
            E[-1] = E[-1] + c / self.epsilon_r[-1] * (2 * H[-1])
        elif self.boundary == "period":
            E[0] += - c_eps[0] * (H[0] - H[-1])
            E[-1] = E[0]
        elif self.boundary == "mur":
            cte = (c-1.0)/(c + 1.0)
            # Left 
            E[0] = E_aux_izq + cte*( E[1] - E[0])
            # Right
            E[-1] = E_aux_dch + cte*( E[-2] - E[-1] )
        else:
            raise ValueError("Boundary not defined")
        
    
    def stepGlobal(self):

        self.dt = self.CFL *np.min(np.array([self.dx, self.sub_dx]))
        E = self.E
        H = self.H
        c = self.dt/self.dx
        c_eps = np.ones(self.epsilon_r.size)
        c_eps[:] = self.dt/self.dx / self.epsilon_r[:]
        E_aux_izq = E[1]
        E_aux_dch= E[-2]

        # H += - self.dt/self.dx *(E[1:] - E[:-1])
        H[0:self.mesh.refined_initial_index] += - self.dt/self.dx *(E[1:self.mesh.refined_initial_index+1] - E[0:self.mesh.refined_initial_index])
        H[self.mesh.refined_initial_index:self.mesh.refined_final_index] += - self.dt/self.sub_dx *(E[self.mesh.refined_initial_index+1:self.mesh.refined_final_index+1] - E[self.mesh.refined_initial_index:self.mesh.refined_final_index])
        H[self.mesh.refined_final_index:] += - self.dt/self.dx *(E[self.mesh.refined_final_index+1:] - E[self.mesh.refined_final_index:-1])

        for source in self.sources:
            H[source.location] += source.function(self.t + self.dt/2)

        E[1:self.mesh.refined_initial_index] += - c_eps[1:self.mesh.refined_initial_index] * (H[1:self.mesh.refined_initial_index] - H[:self.mesh.refined_initial_index-1])
        E[self.mesh.refined_initial_index+1:self.mesh.refined_final_index] += - (self.dx / self.sub_dx) * c_eps[self.mesh.refined_initial_index+1:self.mesh.refined_final_index] * (H[self.mesh.refined_initial_index+1:self.mesh.refined_final_index] - H[self.mesh.refined_initial_index:self.mesh.refined_final_index-1])
        E[self.mesh.refined_initial_index] += - 2 * (self.dt / (self.dx + self.sub_dx))/self.epsilon_r[self.mesh.refined_initial_index] * (H[self.mesh.refined_initial_index] - H[self.mesh.refined_initial_index-1])
        E[self.mesh.refined_final_index] += - 2 * (self.dt / (self.dx + self.sub_dx))/self.epsilon_r[self.mesh.refined_final_index] * (H[self.mesh.refined_final_index] - H[self.mesh.refined_final_index-1])
        E[self.mesh.refined_final_index+1:-1] += - c_eps[self.mesh.refined_final_index+1:-1] * (H[self.mesh.refined_final_index+1:] - H[self.mesh.refined_final_index:-1])
        

        for source in self.sources:
            E[source.location] += source.function(self.t)
        self.t += self.dt

        if self.boundary == "pec":
            E[0] = 0.0
            E[-1] = 0.0
        elif self.boundary == "pmc":
            E[0] = E[0] - c / self.epsilon_r[0] * (2 * H[0])
            E[-1] = E[-1] + c / self.epsilon_r[-1] * (2 * H[-1])
        elif self.boundary == "period":
            E[0] += - c_eps[0] * (H[0] - H[-1])
            E[-1] = E[0]
        elif self.boundary == "mur":
            cte = (c-1.0)/(c + 1.0)
            # Left 
            E[0] = E_aux_izq + cte*( E[1] - E[0])
            # Right
            E[-1] = E_aux_dch + cte*( E[-2] - E[-1] )
        else:
            raise ValueError("Boundary not defined")
        
    def stepLocal(self):

        self.dt = self.CFL *self.dx
        self.sub_dt = self.CFL *self.sub_dx
        E = self.E
        H = self.H
        c = self.dt/self.dx
        c_eps = np.ones(self.epsilon_r.size)
        c_sub_eps = np.ones(self.epsilon_r.size)
        c_eps[:] = self.dt/self.dx / self.epsilon_r[:]
        c_sub_eps[:] = self.sub_dt/self.sub_dx / self.epsilon_r[:]
        E_aux_izq = E[1]
        E_aux_dch= E[-2]

        # H += - self.dt/self.dx *(E[1:] - E[:-1])
        if float(self.t) == 0.0:
            self.Current_Time = self.sub_dt
            self.Previous_Time = 1.0

        self.A = interleaved(self.Current_Time, self.dt)
        if interleaved(self.Current_Time, self.dt) >= self.Current_Time:
            H[0:self.mesh.refined_initial_index] += - self.dt/self.dx *(E[1:self.mesh.refined_initial_index+1] - E[0:self.mesh.refined_initial_index])
            H[self.mesh.refined_final_index:] += - self.dt/self.dx *(E[self.mesh.refined_final_index+1:] - E[self.mesh.refined_final_index:-1])

        H[self.mesh.refined_initial_index:self.mesh.refined_final_index] += - self.sub_dt/self.sub_dx *(E[self.mesh.refined_initial_index+1:self.mesh.refined_final_index+1] - E[self.mesh.refined_initial_index:self.mesh.refined_final_index])


        for source in self.sources:
            H[source.location] += source.function(self.t + self.dt/2)

        if interleaved(self.Current_Time, self.dt) >= self.Current_Time:
            E[1:self.mesh.refined_initial_index] += - c_eps[1:self.mesh.refined_initial_index] * (H[1:self.mesh.refined_initial_index] - H[:self.mesh.refined_initial_index-1])
            E[self.mesh.refined_final_index+1:-1] += - c_eps[self.mesh.refined_final_index+1:-1] * (H[self.mesh.refined_final_index+1:] - H[self.mesh.refined_final_index:-1])
        
        E[self.mesh.refined_initial_index+1:self.mesh.refined_final_index] += - c_sub_eps[self.mesh.refined_initial_index+1:self.mesh.refined_final_index] * (H[self.mesh.refined_initial_index+1:self.mesh.refined_final_index] - H[self.mesh.refined_initial_index:self.mesh.refined_final_index-1])
        E[self.mesh.refined_initial_index] += -  2 * (self.sub_dt / (self.dx + self.sub_dx))/self.epsilon_r[self.mesh.refined_initial_index] * (H[self.mesh.refined_initial_index] - H[self.mesh.refined_initial_index-1])
        E[self.mesh.refined_final_index] += -  2 * (self.sub_dt / (self.dx + self.sub_dx))/self.epsilon_r[self.mesh.refined_final_index] * (H[self.mesh.refined_final_index] - H[self.mesh.refined_final_index-1])

        # for source in self.sources:
        #     E[source.location] += source.function(self.t)
        self.Previous_Time = self.t
        self.t = self.t+self.sub_dt
        self.Current_Time = self.t

        if self.boundary == "pec":
            E[0] = 0.0
            E[-1] = 0.0
        elif self.boundary == "pmc":
            E[0] = E[0] - c / self.epsilon_r[0] * (2 * H[0])
            E[-1] = E[-1] + c / self.epsilon_r[-1] * (2 * H[-1])
        elif self.boundary == "period":
            E[0] += - c_eps[0] * (H[0] - H[-1])
            E[-1] = E[0]
        elif self.boundary == "mur":
            cte = (c-1.0)/(c + 1.0)
            # Left 
            E[0] = E_aux_izq + cte*( E[1] - E[0])
            # Right
            E[-1] = E_aux_dch + cte*( E[-2] - E[-1] )
        else:
            raise ValueError("Boundary not defined")

    
    def run_until(self, finalTime):
        while (self.t <= finalTime):
            if False:    
                plt.plot(self.xE, self.E, '.-')
                plt.plot(self.xH, self.H, '.-')
                plt.ylim(-1.1, 1.1)
                plt.title(self.t)
                plt.grid(which='both')
                plt.pause(0.02)
                plt.cla()
            if self.SolverType is None:
                self.stepDefault()
            elif self.SolverType == 'Global':
                self.stepGlobal()
            elif self.SolverType == 'Local':
                self.stepLocal()
            else:
                raise ValueError('Para hacer estupideces no pongas ningún Solver y que use el predeterminado :/')

class Source():
    def __init__(self, location, function):
        self.location = location
        self.function = function
    def gaussian(location, center, amplitude, spread):
        def function(t):
            return np.exp( - ((t-center)/spread)**2/2) * amplitude
        return Source(location, function)
    def square(location, tini, tfin, amplitude):
        def function(t):
            if t > tini and t < tfin:
                return amplitude 
            else:
                return 0
        return Source(location, function)
          


def test_pec():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01)
    fdtd = FDTD1D(mesh, "pec", CFL = 1.0)

    spread = 0.1
    initialE = np.exp( - ((mesh.xE)/spread)**2/2)

    # plt.plot(mesh.xE, initialE)
    # plt.show()

    fdtd.setE(initialE)
    fdtd.run_until(1.0)

    R = np.corrcoef(fdtd.getE(), -initialE)
    assert np.isclose(R[0,1], 1.0)

def test_pmc():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01)
    fdtd = FDTD1D(mesh, "pmc", CFL=1.0)

    spread = 0.1
    initialE = np.exp( - (mesh.xE/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(1.0)

    R = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R[0,1], 1.0)

def test_period():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01)
    fdtd = FDTD1D(mesh, "period", CFL=1.0)

    spread = 0.1
    initialE = np.exp( - ((mesh.xE-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)


    fdtd.setE(initialE)
    fdtd.run_until(1.0)


    R_E = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R_E[0,1], 1.0, rtol=1.e-2)

    # R_H = np.corrcoef(initialH, fdtd.getH())
    assert np.allclose(fdtd.H, initialH, atol=1.e-2)


def test_pec_dielectric():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01)
    epsilon_r = 4
    epsilon_vector = epsilon_r*np.ones(mesh.xE.size)
    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)

    fdtd = FDTD1D(mesh, "pec", CFL=1.0, relative_epsilon_vector=epsilon_vector)

    spread = 0.1
    initialE = np.exp( - (mesh.xE/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(time)

    R = np.corrcoef(fdtd.getE(), -initialE)
    assert np.isclose(R[0,1], 1.0)

def test_period_dielectric():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01)
    epsilon_r = 4
    epsilon_vector = epsilon_r*np.ones(mesh.xE.size)
    time = np.sqrt(epsilon_r) * np.sqrt(EPSILON_0 * MU_0)
    
    fdtd = FDTD1D(mesh, "period", CFL=1.0, relative_epsilon_vector=epsilon_vector)

    spread = 0.1
    initialE = np.exp( - ((mesh.xE-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)


    fdtd.setE(initialE)
    fdtd.run_until(time)


    R_E = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R_E[0,1], 1.0, rtol=1.e-2)

    # R_H = np.corrcoef(initialH, fdtd.getH())
    assert np.allclose(fdtd.H, initialH, atol=1.e-2)

def test_mur():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01)
    fdtd = FDTD1D(mesh, "mur", CFL=1.0)

    spread = 0.1
    initialE = np.exp( - (mesh.xE/spread)**2/2)

    fdtd.setE(initialE)
    fdtd.run_until(1.1)

    assert np.allclose(fdtd.getE(), np.zeros_like(fdtd.getE()), atol = 1.e-2)

def test_error():
    error = np.zeros(5)
    deltax = np.zeros(5)
    for i in range(5):
        num = 10**(i+1) +1
        mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 1.000/(num))
        deltax[i] = 1/num
        fdtd = FDTD1D(mesh, "pec", CFL=1.0)
        spread = 0.1
        initialE = np.exp( - ((mesh.xE-0.1)/spread)**2/2)
        
        fdtd.setE(initialE)
        fdtd.stepDefault()
        fdtd.stepDefault()
        N = len(initialE)
        error[i] = np.sqrt(np.sum((fdtd.getE() - initialE)**2)) / N
        
    # plt.plot(deltax, error)
    # plt.loglog()
    # plt.grid(which='both')
    # plt.show()
    
    # np.polyfit(np.log10(error), np.log10(deltax), 1)
    
    slope = (np.log10(error[-1]) - np.log10(error[0])) / \
        (np.log10(deltax[-1]) - np.log10(deltax[0]) )


    assert np.isclose( slope , 2, atol=0.13)
                                
def test_illumination():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01)
    fdtd = FDTD1D(mesh, "pec", CFL=1.0)

    fdtd.addSource(Source.gaussian(20, 0.5, 0.5, 0.1))
    fdtd.addSource(Source.gaussian(70, 1.0, -0.5, 0.1))

    fdtd.run_until(1.0)
    assert np.allclose(fdtd.getE()[:20], 0.0, atol = 1e-2)
    assert np.allclose(fdtd.getE()[71:], 0.0, atol = 1e-2)
    fdtd.run_until(3.0)
    assert np.allclose(fdtd.getE(), 0.0, atol = 1e-2)

def test_visual_subgriding_mesh():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01, sub_dx = 0.004, refined_initial_index = 10, refined_final_index = 40)
    # plt.plot(mesh.xH, np.zeros(mesh.xH.size), 'o-')
    # plt.grid()
    # plt.show()
    for i in range(mesh.refined_initial_index - 1):
        assert np.isclose(np.abs(mesh.xE[i]-mesh.xE[i+1]), mesh.dx)

    for i in range(mesh.refined_initial_index, mesh.refined_final_index-1):
        assert np.isclose(np.abs(mesh.xE[i]-mesh.xE[i+1]), mesh.sub_dx)
        


def test_pec_subgriding():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01, sub_dx = 0.001, refined_initial_index = 10, refined_final_index = 40)
    fdtd = FDTD1D(mesh, "pec", CFL = 0.8,  SolverType='Global')

    spread = 0.1
    initialE = np.exp( - ((mesh.xE)/spread)**2/2)

    # plt.plot(mesh.xE, initialE)
    # plt.show()

    fdtd.setE(initialE)
    fdtd.run_until(1.0)

    R = np.corrcoef(fdtd.getE(), -initialE)
    assert np.isclose(R[0,1], 1.0, atol=1e-3)

def test_period_subgriding():
    mesh = Mesh(initial_position= -0.5, final_position = 0.5, dx = 0.01, sub_dx = 0.005, refined_initial_index = 10, refined_final_index = 40)
    fdtd = FDTD1D(mesh, "period", CFL = 0.8,  SolverType='Global')

    spread = 0.1
    initialE = np.exp( - ((mesh.xE-0.1)/spread)**2/2)
    initialH = np.zeros(fdtd.H.shape)


    fdtd.setE(initialE)
    fdtd.run_until(1.0)


    R_E = np.corrcoef(fdtd.getE(), initialE)
    assert np.isclose(R_E[0,1], 1.0, rtol=1.e-2)
