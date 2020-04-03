import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.patches import Rectangle

class VizEnv():
    labels = {0:'Susceptible', 1:'Infectious', 2:'Recovered'}

    def __init__(self, n_rows, n_columns, regions, step_f):
        self.main_figure = plt.figure(num = 0, figsize = (12, 8), constrained_layout=True)
        self.main_figure.suptitle("Environment", fontsize=12)

        self.n_rows = n_rows
        self.n_columns = n_columns

        self.step_f = step_f
        self.regions = regions

        self.axs = self.main_figure.subplots(n_rows, n_columns)
        self.p = [[0]*n_columns for _ in range(n_rows)]

        for i, reg in enumerate(regions.values()):
            pos = reg['population'][:, :2].T
            state = reg['population'][:, 2]

            
            col = i % n_columns
            row = i // n_columns
            
            self.axs[row,col].set_title("Region " + reg['name'])
            self.axs[row, col].set_xlim(-1, reg['size']+1)
            self.axs[row, col].set_ylim(-1, reg['size']+1)
            #self.axs[row, col].grid()

            self.axs[row, col].add_patch(Rectangle((0,0), reg['size'], reg['size'], alpha=0.1))
            
            self.p[row][col] = self.axs[row,col].scatter(*pos, s=10, c=state)

        #self.main_figure.legend()
        
    def update(self, index):
        self.regions = self.step_f(self.regions)
        for i, reg in enumerate(self.regions.values()):
            pos = reg['population'][:, :2]
            state = reg['population'][:, 2]
            
            col = i % self.n_columns
            row = i // self.n_columns
            self.p[row][col].set_offsets(pos)
            self.p[row][col].set_array(state)

    def show(self, frames):
        simulation = animation.FuncAnimation(self.main_figure, self.update, frames=frames)
        plt.show()

def init_plots():
    global p, axs, n_rows, n_columns, main_figure
    main_figure = plt.figure(num = 0, figsize = (12, 8))#, dpi = 100)
    main_figure.suptitle("Environment", fontsize=12)

    # initialize plots
    n_rows = 4
    n_columns = int(np.ceil(len(regions)/n_rows))

    axs = main_figure.subplots(n_rows, n_columns)
    p = [[0]*n_columns for _ in range(n_rows)]

    for i, reg in enumerate(regions.values()):
        pos = reg['population'].T
        
        col = i % n_columns
        row = i // n_columns
        
        axs[row,col].set_title("Region " + reg['name'])
        p[row][col] = axs[row,col].scatter(*pos, s=1)
    
n_regions = 10 


def init_regions(n_regions):
    regions = dict()
    for reg in range(n_regions):
        init_population = np.random.randint(low=10 , high=20)
        region_size = np.random.uniform(5,10)
        population = np.zeros((init_population, 3))
        population[:, :2] = np.random.uniform(low=0, high=region_size, size=(init_population, 2))
        population[:, 2] = np.random.randint(low=0, high=3, size=(init_population)) # state
        regions[reg] = {'population' : population, 'name': str(reg), 'size': region_size}
        print(reg, region_size)
    return regions

def move(population, size):
    n_population = population.shape[0]

    movement = get_moves(n_population)
    tentative_movement = population + movement # try movement to see if we get

    pos = np.sum(((tentative_movement > size) + (tentative_movement < 0) != 0), axis=1)
    inside = pos == 0

    population =  tentative_movement
    population[(tentative_movement > size)] = size
    population[(tentative_movement < 0)] = 0


    # inside = pos == 0
    # outside = pos == 1
    # print(inside.shape, population.shape)
    # print(tentative_movement.shape)

    # #population[inside,:] = tentative_movement[inside,:]
    
    # while outside.sum() != 0:
    #     movement_outside = get_moves(outside.sum())

    #     print(movement[outside,:].shape, movement_outside.shape)


    #     movement[outside,:] = movement_outside
        
    #     tentative_movement = population + movement
        
    #     pos = np.sum(((tentative_movement > size) + (tentative_movement < 0) != 0), axis=1)
    #     inside = pos == 0
    #     outside = pos == 1

    #     population[inside,:] = tentative_movement[inside,:]


    # vec_rotation = (-1*(tentative_movement > size)) + (-1*(tentative_movement < 0))


    return population


def get_moves(n_moves):
    movement = np.random.uniform(low=-1, high=1, size=(n_moves, 2))
        
    normalization = np.expand_dims(np.sqrt(np.sum(movement**2, axis=1)), axis=1)   
    movement = movement/normalization

    magnitude = np.random.uniform(low=0.01, high=1.0, size=(n_moves, 1))

    return movement*magnitude
        

def step(regions):
    magnitudes = .1
    for reg_k, reg_v in regions.items():
        n_population = reg_v['population'].shape[0]
        
        # outside
        population = reg_v['population'][:, :2]
        state = reg_v['population'][:, 2]

        population = move(population, reg_v['size'])
        #state = update_states()

        regions[reg_k]['population'][:, :2] = population
        regions[reg_k]['population'][:,2] = state

    return regions

regions = init_regions(n_regions)

n_rows = 4
n_columns = int(np.ceil(len(regions)/n_rows))

vis = VizEnv(n_rows, n_columns, regions, step)
vis.show(100)