
import numpy as np
import json
from viz import VizEnv


class Simulator():
    def __init__(self, population, names, sizes, region_first_infected, infection_radius, hygiene=0.2):
        self.regions = dict()

        self.hygiene = hygiene
        self.infection_radius = infection_radius

        for r in range(len(population)):
            n_small = 20
            n_medium = 2
            n_big = 1
            n_total = n_small + n_medium + n_big

            mean = np.random.uniform(low=0, high=sizes[r], size=(n_total, 2))

            surface_population =  np.sqrt(population[r]/sizes[r])

            # more variance more spreading = less concentration = p(infection) lower
            d_small = surface_population*10
            d_medium = surface_population*1
            d_big = surface_population*0.5

            density = np.array([d_small]*n_small + [d_medium]*n_medium + [d_big]*n_big)
            
            pos = np.zeros((population[r], 3))
            if r == region_first_infected:
                pos = np.zeros((population[r]+1, 3))

            self.regions[r] = {
                               'S': population[r], 
                               'I':0, 
                               'R':0, 
                               'name': names[r], 
                               'size': sizes[r],
                               'centers': mean,
                               'density': density,
                               'policy': np.ones(n_total)/n_total, # init = uniform
                               'isolated': False,
                               'pos': pos
                            }
                    
        # add infected
        self.regions[region_first_infected]['I'] = 1
        self.get_contourf()

    def mixture(self, means, sigmas, probs, n_samples):
        data = np.zeros((n_samples, 2))
        # Chose gaussians (maybe we can just take %p of samples of each type of gaussian)
        gaussians = np.random.choice(len(means), n_samples, p=probs)
        for i, g in enumerate(gaussians):
            cov = np.eye(2)*sigmas[g]
            data[i,:] = np.random.multivariate_normal(means[g], cov, 1)
        
        return data

    def normal_pdf(self, X, Y, mean, sigma):
        dim = mean.shape[0]

        k = 1 #(np.sqrt(np.pi*2)*sigma)
        norm_x = np.exp(-((X-mean[0])**2)/(2*sigma**2))/k
        norm_y = np.exp(-((Y-mean[1])**2)/(2*sigma**2))/k

        return norm_x*norm_y

    def mixture_pdf(self, X, Y, mean, std, probs):
        mix = np.zeros(X.shape)
        for i, p in enumerate(probs):
            mix += p*self.normal_pdf(X, Y, mean[i,:], std[i])
        
        return mix

    def get_contourf(self):
        for reg_k, reg_v in self.regions.items():
            self.regions[reg_k]['mixture'] = dict()

            N = 400
            X = np.linspace(0, reg_v['size'], N)
            Y = np.linspace(0, reg_v['size'], N)
            X, Y = np.meshgrid(X, Y)
            
            mean = reg_v['centers']
            std = reg_v['density']
            probs = reg_v['policy']

            self.regions[reg_k]['mixture']['X'] = X
            self.regions[reg_k]['mixture']['Y'] = Y
            self.regions[reg_k]['mixture']['Z'] = self.mixture_pdf(X, Y, mean, std, probs)

    def step(self):
        for reg_k, reg_v in self.regions.items():
            S = self.mixture(reg_v['centers'], reg_v['density'], reg_v['policy'], self.regions[reg_k]['S'])
            I = self.mixture(reg_v['centers'], reg_v['density'], reg_v['policy'], self.regions[reg_k]['I'])
            R = self.mixture(reg_v['centers'], reg_v['density'], reg_v['policy'], self.regions[reg_k]['R'])
            
            for i in range(I.shape[0]):
                dist = np.sqrt(np.sum((S - I[i,:])**2, axis=1))

                contacted = dist < self.infection_radius

                dice = np.random.random(contacted.shape)
                infected = (dice < self.hygiene)*contacted

                infected_individuals = S[infected, :]
                I = np.concatenate([I, infected_individuals])
                S = S[np.logical_not(infected), :]

                self.regions[reg_k]['S'] -= np.sum(infected)
                self.regions[reg_k]['I'] += np.sum(infected)
            
            pos = np.concatenate([S, I, R])
            state = np.array([[0]* self.regions[reg_k]['S'] + [1]*self.regions[reg_k]['I'] + [2]*self.regions[reg_k]['R']]).T
            self.regions[reg_k]['pos'] = np.hstack((pos, state))


            # transition to other regions (if allowed)
            if not reg_v['isolated']:
                max_travelers = int((self.regions[reg_k]['S'] + self.regions[reg_k]['I'] + self.regions[reg_k]['R'])*0.01) # only 10% 
                travelers = np.random.randint(low=0, high=max(max_travelers, 0)) # max travelers
                label = ['S', 'I', 'R']

                perc = np.array([self.regions[reg_k]['S'], self.regions[reg_k]['I'], self.regions[reg_k]['R']])
                
                norm =  np.sum(perc) if np.sum(perc) != 0 else 1.0
                perc = perc / norm
                
                if travelers > 0:
                    traveller_status =  [int(p*travelers) for p in perc]# np.random.choice(3, travelers, p=p)
                    for i in range(3):
                        s = label[i]
                        travel_to = np.random.randint(low=0, high=len(self.regions))
                        self.regions[travel_to][s] += traveller_status[i]
                        self.regions[reg_k][s] -= traveller_status[i]
            
            for reg_k, reg_v in self.regions.items():
                print('\n', reg_v['name'], reg_v['S'], reg_v['I'], reg_v['R'])
                print(reg_v['pos'][:,2].mean())

        return self.regions

if __name__ == '__main__':
    population = [700, 100, 100, 100]
    names = ['CAT', 'MAD', 'AND', 'RIOJA']
    sizes = [90, 100, 90, 100]

    sim = Simulator(population, names, sizes, 0, 1, 0.3)
    sim.step()

    n_rows = 2
    n_columns = 2 #int(np.ceil(len(population)/n_rows))

    vis = VizEnv(n_rows, n_columns, sim.regions, sim.step)
    vis.show(100)


