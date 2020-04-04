import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.patches import Rectangle

import numpy as np

class VizEnv():
    color = {0: [0, 1, 0], 1: [1, 0, 0], 2: [0,0,1]}

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
            pos = reg['pos'][:, :2].T
            state = reg['pos'][:, 2]

            col = i % n_columns
            row = i // n_columns

            self.axs[row,col].set_title("Region " + reg['name'])
            self.axs[row, col].set_xlim(-1, reg['size']+1)
            self.axs[row, col].set_ylim(-1, reg['size']+1)
            #self.axs[row, col].grid()

            self.axs[row, col].add_patch(Rectangle((0,0), reg['size'], reg['size'], alpha=1.0, fill=False, color='black', lw=5))
            
            self.axs[row, col].contourf(reg['mixture']['X'], reg['mixture']['Y'], reg['mixture']['Z'])

            self.p[row][col] = self.axs[row,col].scatter(*pos, s=10, c=[VizEnv.color[s] for s in state])
        
    def update(self, index):
        self.regions = self.step_f() # self.step_f(regions)
        for i, reg in enumerate(self.regions.values()):
            pos = reg['pos'][:, :2]
            state = reg['pos'][:, 2]
            
            col = i % self.n_columns
            row = i // self.n_columns
            self.p[row][col].set_offsets(pos)
            self.p[row][col].set_color([VizEnv.color[s] for s in state])

    def show(self, frames):
        simulation = animation.FuncAnimation(self.main_figure, self.update, frames=frames)
        plt.show()