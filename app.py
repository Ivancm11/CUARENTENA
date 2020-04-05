from flask import Flask, redirect, url_for, render_template, request,jsonify
import numpy as np
import json
import os
os.environ["FLASK_ENV"]="deployment"

app = Flask(__name__, static_folder='static', static_url_path='')

class Simulator():
    def __init__(self, population, names, sizes, UCI, region_first_infected, infection_radius, hygiene=0.2):
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
                               'uci': UCI[r],
                               'size': sizes[r],
                               'centers': mean,
                               'density': density,
                               'policy': np.ones(n_total)/n_total, # init = uniform
                               'isolated': False,
                               'pos': pos
                            }

            if r == region_first_infected:
                self.regions[r]['infection_time'] = np.array([0])
                    
        # add infected
        self.regions[region_first_infected]['I'] = 1
        self.get_contourf()
        self.i = 0

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
            
            n_infected = 0

            for i in range(I.shape[0]):
                dist = np.sqrt(np.sum((S - I[i,:])**2, axis=1))

                contacted = dist < self.infection_radius

                dice = np.random.random(contacted.shape)
                infected = (dice < self.hygiene)*contacted

                infected_individuals = S[infected, :]
                I = np.concatenate([I, infected_individuals])
                S = S[np.logical_not(infected), :]
                
                n_infected += sum(infected)

            self.regions[reg_k]['S'] -= n_infected #np.sum(infected)
            self.regions[reg_k]['I'] += n_infected #np.sum(infected)

            
            
            if 'infection_time' not in self.regions[reg_k].keys():
                self.regions[reg_k]['infection_time'] = np.zeros(n_infected)
            else:
                self.regions[reg_k]['infection_time'] = np.concatenate([self.regions[reg_k]['infection_time'], np.zeros(n_infected)])
            self.regions[reg_k]['infection_time'] += 1

            steps_removed = 17
            cured_or_dead = self.regions[reg_k]['infection_time'] > steps_removed

            n_removed = np.sum(cured_or_dead)

            self.regions[reg_k]['infection_time'] = self.regions[reg_k]['infection_time'][np.logical_not(cured_or_dead)]

            self.regions[reg_k]['R'] += n_removed
            self.regions[reg_k]['I'] -= n_removed
            
            if R.shape[0] == 0 :
                R = I[:n_removed,:]
            else:
                R = np.concatenate([R,  I[:n_removed,:]])

            I = I[n_removed:, :]

            pos = np.concatenate([S, I, R])
            state = np.array([[0]* self.regions[reg_k]['S'] + [1]*self.regions[reg_k]['I'] + [2]*self.regions[reg_k]['R']]).T
            self.regions[reg_k]['pos'] = np.hstack((pos, state))

            # transition to other regions (if allowed)
            if not reg_v['isolated']:
                max_travelers = int((self.regions[reg_k]['S'] + self.regions[reg_k]['I'] + self.regions[reg_k]['R'])*0.01) # only 1% 
                travelers = np.random.randint(low=0, high=max_travelers) # max travelers
                label = ['S', 'I', 'R']

                perc = np.array([self.regions[reg_k]['S'], self.regions[reg_k]['I'], self.regions[reg_k]['R']])
                
                norm =  np.sum(perc) if np.sum(perc) != 0 else 1.0
                perc = perc / norm
                
                if travelers > 0:
                    traveller_status =  [int(p*travelers) for p in perc] # np.random.choice(3, travelers, p=p)
                    for i in range(3):
                        s = label[i]
                        travel_to = np.random.randint(low=0, high=len(self.regions))
                        
                        self.regions[travel_to][s] += traveller_status[i]
                        self.regions[reg_k][s] -= traveller_status[i]
                        
                        # move to the infectious time queue of travel_to
                        if s == 'I':
                            if 'infection_time' not in self.regions[travel_to].keys():
                                self.regions[travel_to]['infection_time'] = self.regions[reg_k]['infection_time'][:traveller_status[i]]
                            else:
                                self.regions[travel_to]['infection_time'] = np.concatenate([self.regions[travel_to]['infection_time'], self.regions[reg_k]['infection_time'][:traveller_status[i]]])

                            self.regions[reg_k]['infection_time'] = self.regions[reg_k]['infection_time'][traveller_status[i]:]
                
        total_S = 0
        total_I = 0
        total_R = 0
        self.i += 1
        return self.regions

    def get_state(self):
        state = []
        for reg_v in self.regions.values():
            state = state + [reg_v['S'], reg_v['I'], reg_v['R']]
        return state

    def get_reward_terminal(self):
        r = 0
        terminal = 0

        size_isolated = sum([reg_v['isolated']*reg_v['size'] for reg_v in self.regions.values()])

        uci = np.array([reg_v['uci'] for reg_v in self.regions.values()])
        infected = np.array([reg_v['uci'] for reg_v in self.regions.values()])

        mortality = np.mean(infected/uci) # a function of remaining UCI

        if sum([reg_v['I'] for reg_v in self.regions.values()]) == 0:
            terminal = 1
            r = 100


        r += mortality + np.mean(infected) + size_isolated

        return r, terminal

    def simulate(self, agent):
        total_S = 0
        total_I = 0
        total_R = 0

        regions_resumed = dict()

        # init
        for i, reg_v in enumerate(self.regions.values()):
            regions_resumed[i] = {
                'name':reg_v['name'],
                'size':reg_v['size'],
                'S': [reg_v['S']],
                'I': [reg_v['I']],
                'R': [0],
                'pos': {
                    'x':[],
                    'y':[],
                    'state':[]
                }
            }

        while total_R == 0 or total_I != 0:
            total_S = 0
            total_I = 0
            total_R = 0

            state = self.get_state()
            reward, terminal = self.get_reward_terminal()


            actions = agent(state) # self.hygiene (0), self.radius, isolation (for region), p (region)
                                   # state self.S, self.I, self.R for regions

            # TODO: Modify params

            self.step()

            nex_state = self.get_state()

            for i, reg_v in enumerate(self.regions.values()):
                
                total_S += reg_v['S']
                total_I += reg_v['I']
                total_R += reg_v['R']
                
                for k in ('S', 'I', 'R'):
                    regions_resumed[i][k].append(int(reg_v[k]))

                regions_resumed[i]['pos']['x'].append(reg_v['pos'][:,0].tolist())
                regions_resumed[i]['pos']['y'].append(reg_v['pos'][:,1].tolist())
                regions_resumed[i]['pos']['state'].append(reg_v['pos'][:,2].tolist())


        
        # plt.plot(regions_resumed[0]['S'], c='green')
        # plt.plot(regions_resumed[0]['I'], c='red')
        # plt.plot(regions_resumed[0]['R'], c='black')
        # plt.legend(['Susceptible', 'Infected', 'Removed'])
        # plt.savefig('curve.png', dpi=300)
        # plt.show()
        

        return regions_resumed

def agent(state):
    return []

# if __name__ == '__main__':
#     population = [700, 100, 100, 100]
#     names = ['CAT', 'MAD', 'AND', 'RIOJA']
#     sizes = [90, 100, 90, 100]
    
#     UCI = [9, 10, 9, 10]

#     sim = Simulator(population, names, sizes, UCI, 0, 1, 0.3)
#     sim.step()
#     json_str = sim.simulate(agent)
    



def model():
    population = [50, 50, 50]
    names = ['CAT', 'MAD', 'AND']
    sizes = [90, 100, 90]
    
    UCI = [9, 10, 9]


    sim = Simulator(population, names, sizes, UCI, 0, 1, 0.3)
    sim.step()
    json_str = sim.simulate(agent)
    return json_str
   

    



@app.route("/", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        percentage_aux = request.form["percentage"]
        capacity_aux=request.form["capacity"]
        min_infected_aux=request.form["min_infected"]
        json_data=model()
        Susceptible=np.zeros(len(json_data[0]['S']))
        Infected=np.zeros(len(json_data[0]['I']))
        Recovered=np.zeros(len(json_data[0]['R']))
        for region in json_data.keys():
            Susceptible+=json_data[region]['S']
            Infected+=json_data[region]['I']
            Recovered+=json_data[region]['R']
        Susceptible=list(Susceptible)
        Infected=list(Infected)
        Recovered=list(Recovered)




        return render_template('index.html',data_aux=json_data,susceptible=Susceptible,infected=Infected,recovered=Recovered)
    else:
        return render_template("inputs.html")




if __name__ == "__main__":
    app.run(debug=False)