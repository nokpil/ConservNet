import numpy as np
import math
import random

def sign():
    return 1 if np.random.random() < 0.5 else -1

# S1 : x-2yz+3w^2
def system_S1(batch_size, batch_num):
    CONSTANT = -5.
    counter = 0
    while(True):
        if counter%batch_size == 0 :
            CONSTANT += 10/batch_num
            counter = 0
        counter+=1
        while True:
            data = np.random.normal(0, 2, 4)
            data[0] = CONSTANT + 3*data[1]*data[2] - 0.5*data[3]**2
            if np.abs(data[0])<5:
                break
        answer = CONSTANT # Answer is not fixed, thus meaningless in this scheme
        yield data, answer

# S2 : 3x + 2*sin(y) + sqrt(x)z^2
def system_S2(batch_size, batch_num):
    CONSTANT = np.random.uniform(-5, 5)
    CONSTANT = -5.
    counter = 0
    offset = [-2*np.pi, 0, 2*np.pi]
    while(True):
        if counter%batch_size == 0 :
            #CONSTANT = np.random.uniform(-5, 5)
            CONSTANT += 5/batch_num
            counter = 0
            
        counter+=1
    
        while (True):
            x1 = np.random.uniform(-3, 3)
            fx1 = 3*x1
            x3 = np.random.uniform(-3, 3)
            if np.abs(CONSTANT-fx1-np.sqrt(np.abs(x1))*(x3**3))/2 <= 1:
                break
          
        x2 = np.arcsin((CONSTANT-fx1-np.sqrt(np.abs(x1))*(x3**3))/2)

        if x2 > 0 and np.random.random()>0.5:
                x2 = np.pi-x2
        if x2 < 0 and np.random.random()>0.5:
            x2 = -np.pi-x2

        x2 = x2 +offset[np.random.randint(0,len(offset))]
        assert np.abs(np.sin(x2)-(CONSTANT-fx1-np.sqrt(np.abs(x1))*(x3**3))/2) < 1e-6

        data = np.random.normal(0, 1, 3)
        data[0] = x1
        data[1] = x2
        data[2] = x3
        #data = np.array([x1,x2])
        answer = CONSTANT # Answer is not fixed, thus meaningless in this scheme
        yield data, answer

# S3: log / rational
def system_S3(batch_size, batch_num):
    CONSTANT = 1.
    counter = 0
    while(True):
        counter+=1
        if counter%batch_size == 0:
            CONSTANT += 3/batch_num
            counter = 0
        while True:
            data = np.random.uniform(-10, 10, 4)
            data[3] = np.random.uniform(0.5, 5)
            data[0] = np.log(np.abs(data[1]+data[3]))-(2*data[1]*data[2]-CONSTANT)*data[3]
            if np.abs(data[0]) < 10:
                break
        answer = CONSTANT # Answer is not fixed, thus meaningless in this scheme
        yield data, answer

# P1 : Lotka-Volettra system
def system_P1(batch_size, batch_num):
    counter = -1
    alpha, beta, delta, gamma = 0.11, 0.04, 0.01, 0.04
    while(True):
        counter+=1
        if counter%batch_size == 0 :
            print(counter)
            counter=0
            R0 = np.random.uniform(1, 10)
            F0 = np.random.uniform(1, 10)
            jump = 1000
            CONSTANT = alpha * np.log(F0) + beta * np.log(R0) - gamma * F0 - delta* R0
            t = np.linspace(0, 4*batch_size/10, (jump*batch_size)+1)
            R, F = LotkaVolterra_EEuler(R0, F0, alpha, beta, gamma, delta, t)
            R, F, t = R[::jump], F[::jump], t[::jump]
            data = np.concatenate((R.reshape(-1, 1),F.reshape(-1, 1)), axis=-1)
            answer = CONSTANT
            C = alpha * np.log(data[:,1]) + beta * np.log(data[:,0]) - gamma * data[:,1] - delta* data[:,0]
            
        yield data[counter], answer

# P1 : Kepler problem
def system_P2(batch_size, batch_num):
    counter = -1
    G_const = 6.67408e-11
    while(True):
        counter+=1
        if counter%batch_size == 0 :
            print(counter)
            while (True):
                loc1 = np.random.uniform(-5, 5) * sign()
                loc2 = np.random.uniform(-5, 5) * sign()
                vel1 = np.random.uniform(-5, 5) * sign()
                vel2 = np.random.uniform(-5, 5) * sign()
                counter = 0

                sun = {"location":point(0,0,0), "mass":1/G_const, "velocity":point(0,0,0)}
                earth = {"location":point(loc1,loc2,0), "mass":1, "velocity":point(vel1,vel2,0)}

                #build list of planets in the simulation, or create your own
                bodies = [
                    body( location = sun["location"], mass = sun["mass"], velocity = sun["velocity"], name = "sun"),
                    body( location = earth["location"], mass = earth["mass"], velocity = earth["velocity"], name = "earth"),
                    ]
                burn = 3
                #motions = run_simulation(bodies, time_step = 0.01, number_of_steps = 40000+1, report_freq = int(40000/400))
                motions = run_simulation(bodies, time_step = 0.01, number_of_steps = (100*(batch_size+burn))+1, report_freq = 100)
                
                data = np.random.normal(0, 1, (batch_size, 4))
                x = np.array(motions[1]['x'][burn:])
                y = np.array(motions[1]['y'][burn:])
                vx = np.array(motions[1]['vx'][burn:])
                vy = np.array(motions[1]['vy'][burn:])

                a1 = x*vy - y*vx  # Angular momentum
                a2 = 0.5*(np.power(vx,2)+np.power(vy,2)) - 1/np.sqrt(np.power(x,2)+np.power(y,2))  # Total energy
                CONSTANT1 = np.mean(a1)
                CONSTANT2 = np.mean(a2)

                data[:, 0], data[:, 1], data[:, 2] ,data[:, 3] = x, y, vx, vy
                answer = [CONSTANT1, CONSTANT2] # Answer is not fixed, thus meaningless in this scheme
                
                if np.std(a2) < 1e-3 and np.std(a2)<1e-3:
                    ecc = np.sqrt(1+2*(np.mean(a1)**2)*np.mean(a2))
                    #print(ecc)
                    if ecc < 0.99:
                        break
            #print('changed', CONSTANT1, CONSTANT2)
            
        yield data[counter], answer        

# P3 : Real double pendulum system
def plugin_P3(batch_size, batch_num):
    with open('./data/Real_dp.txt', 'rb') as f:
        data_dp = pd.read_csv(f, sep=' ')
    data_dp = data_dp[data_dp['trial']==0]
    data_dp = data_dp[['t1', 't2', 'o1', 'o2']].values
    index=0
    counter=0
    while(True):
        current_data = data_dp
        counter+=1
        data = current_data[counter]
        answer = index  # We don't know!
        yield data, answer

def LotkaVolterra_EEuler(R0, F0, alpha, beta, gamma, delta, t):
    R = np.zeros(len(t)) # Pre-allocate the memory for R
    F = np.zeros(len(t)) # Pre-allocate the memory for F
    R[0] = R0
    F[0] = F0
    for n in range(0,len(t)-1):
        dt = t[n+1] - t[n]
        R[n+1] = R[n]*(1 + alpha*dt - gamma*dt*F[n])
        F[n+1] = F[n]*(1 - beta*dt + delta*dt*R[n])
    return R, F
class point:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z
class body:
    def __init__(self, location, mass, velocity, name = ""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name

def calculate_single_body_acceleration(bodies, body_index):
    G_const = 6.67408e-11 #m3 kg-1 s-2
    #G_const = 1
    acceleration = point(0,0,0)
    target_body = bodies[body_index]
    for index, external_body in enumerate(bodies):
        if index != body_index:
            r = (target_body.location.x - external_body.location.x)**2 + (target_body.location.y - external_body.location.y)**2 + (target_body.location.z - external_body.location.z)**2
            r = math.sqrt(r)
            tmp = G_const * external_body.mass / r**3
            acceleration.x += tmp * (external_body.location.x - target_body.location.x)
            acceleration.y += tmp * (external_body.location.y - target_body.location.y)
            acceleration.z += tmp * (external_body.location.z - target_body.location.z)

    return acceleration

def compute_velocity(bodies, time_step = 1):
    for body_index, target_body in enumerate(bodies):
            acceleration = calculate_single_body_acceleration(bodies, body_index)

            target_body.velocity.x += acceleration.x * time_step
            target_body.velocity.y += acceleration.y * time_step
            target_body.velocity.z += acceleration.z * time_step 

def update_location(bodies, time_step = 1):
    for target_body in bodies:
        if target_body.name!='sun':
            target_body.location.x += target_body.velocity.x * time_step
            target_body.location.y += target_body.velocity.y * time_step
            target_body.location.z += target_body.velocity.z * time_step

def compute_gravity_step(bodies, time_step = 1):
    compute_velocity(bodies, time_step = time_step)
    update_location(bodies, time_step = time_step)

def plot_output(bodies, outfile = None):
    fig = plt.figure()
    colours = ['r','b','g','y','m','c']
    ax = fig.add_subplot(1,1,1, projection='3d')
    max_range = 0
    for i, current_body in enumerate(bodies): 
        max_dim = max(max(current_body["x"]),max(current_body["y"]),max(current_body["z"]))
        if max_dim > max_range:
            max_range = max_dim
        ax.plot(current_body["x"], current_body["y"], current_body["z"], c = colours[i%len(colours)], label = current_body["name"], marker = 'o', markersize = 3, linewidth = 1)        
        ax.plot(current_body["x"][-1], current_body["y"][-1], current_body["z"][-1], c = colours[i%len(colours)],marker = 'o', markersize = 10) 
    #ax.set_xlim([-max_range,max_range])    
    #ax.set_ylim([-max_range,max_range])
    #ax.set_zlim([-max_range,max_range])
    ax.legend()        

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

def run_simulation(bodies, names = None, sun=True, time_step = 1, number_of_steps = 10000, report_freq = 100):

    #create output container for each body
    body_locations_hist = []
    for current_body in bodies:
        body_locations_hist.append({"m":[], "x":[], "vx":[], "y":[], "vy":[], "z":[], "vz":[], "name":current_body.name})
        
    for i in range(1,number_of_steps):
        
        compute_gravity_step(bodies, time_step = time_step)
        
        if i % report_freq == 0:
            for index, body_location in enumerate(body_locations_hist):
                body_location["m"].append(bodies[index].mass)
                
                body_location["x"].append(bodies[index].location.x)
                body_location["y"].append(bodies[index].location.y)           
                body_location["z"].append(bodies[index].location.z)
                
                body_location["vx"].append(bodies[index].velocity.x)
                body_location["vy"].append(bodies[index].velocity.y)           
                body_location["vz"].append(bodies[index].velocity.z)
                
    return body_locations_hist     
