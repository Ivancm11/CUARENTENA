from flask import Flask, redirect, url_for, render_template, request, jsonify
import numpy as np
import json
import os

os.environ["FLASK_ENV"] = "deployment"

app = Flask(__name__, static_folder='static', static_url_path='')


class Simulator():
    def __init__(self, population, names, sizes, UCI, region_first_infected, infection_radius, hygiene=0.5):
        self.regions = dict()

        self.hygiene = hygiene
        self.infection_radius = infection_radius

        for r in range(len(population)):
            n_small = 20
            n_medium = 2
            n_big = 1
            n_total = n_small + n_medium + n_big

            mean = np.random.uniform(low=0, high=sizes[r], size=(n_total, 2))

            surface_population = np.sqrt(population[r] / sizes[r])

            # more variance more spreading = less concentration = p(infection) lower
            d_small = surface_population * 10
            d_medium = surface_population * 1
            d_big = surface_population * 0.5

            density = np.array([d_small] * n_small + [d_medium] * n_medium + [d_big] * n_big)

            pos = np.zeros((population[r], 3))
            if r == region_first_infected:
                pos = np.zeros((population[r] + 1, 3))

            self.regions[r] = {
                'S': population[r],
                'I': 0,
                'R': 0,
                'name': names[r],
                'uci': UCI[r],
                'size': sizes[r],
                'centers': mean,
                'density': density,
                'policy': np.ones(n_total) / n_total,  # init = uniform
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
            cov = np.eye(2) * sigmas[g]
            data[i, :] = np.random.multivariate_normal(means[g], cov, 1)

        return data

    def normal_pdf(self, X, Y, mean, sigma):
        dim = mean.shape[0]

        k = 1  # (np.sqrt(np.pi*2)*sigma)
        norm_x = np.exp(-((X - mean[0]) ** 2) / (2 * sigma ** 2)) / k
        norm_y = np.exp(-((Y - mean[1]) ** 2) / (2 * sigma ** 2)) / k

        return norm_x * norm_y

    def mixture_pdf(self, X, Y, mean, std, probs):
        mix = np.zeros(X.shape)
        for i, p in enumerate(probs):
            mix += p * self.normal_pdf(X, Y, mean[i, :], std[i])

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
                dist = np.sqrt(np.sum((S - I[i, :]) ** 2, axis=1))

                contacted = dist < self.infection_radius

                dice = np.random.random(contacted.shape)
                infected = (dice < self.hygiene) * contacted

                infected_individuals = S[infected, :]
                I = np.concatenate([I, infected_individuals])
                S = S[np.logical_not(infected), :]

                n_infected += sum(infected)

            self.regions[reg_k]['S'] -= n_infected  # np.sum(infected)
            self.regions[reg_k]['I'] += n_infected  # np.sum(infected)

            if 'infection_time' not in self.regions[reg_k].keys():
                self.regions[reg_k]['infection_time'] = np.zeros(n_infected)
            else:
                self.regions[reg_k]['infection_time'] = np.concatenate(
                    [self.regions[reg_k]['infection_time'], np.zeros(n_infected)])

            self.regions[reg_k]['infection_time'] += 1

            steps_removed = 20
            cured_or_dead = self.regions[reg_k]['infection_time'] > steps_removed

            n_removed = np.sum(cured_or_dead)

            self.regions[reg_k]['infection_time'] = self.regions[reg_k]['infection_time'][np.logical_not(cured_or_dead)]

            self.regions[reg_k]['R'] += n_removed
            self.regions[reg_k]['I'] -= n_removed

            if R.shape[0] == 0:
                R = I[:n_removed, :]
            else:
                R = np.concatenate([R, I[:n_removed, :]])

            I = I[n_removed:, :]

            pos = np.concatenate([S, I, R])
            state = np.array(
                [[0] * self.regions[reg_k]['S'] + [1] * self.regions[reg_k]['I'] + [2] * self.regions[reg_k]['R']]).T
            self.regions[reg_k]['pos'] = np.hstack((pos, state))

            # transition to other regions (if allowed)
            if not reg_v['isolated']:
                max_travelers = int(
                    (self.regions[reg_k]['S'] + self.regions[reg_k]['I'] + self.regions[reg_k]['R']) * 0.05)  # only 1%
                travelers = np.random.randint(low=0, high=max_travelers)  # max travelers
                label = ['S', 'I', 'R']

                perc = np.array([self.regions[reg_k]['S'], self.regions[reg_k]['I'], self.regions[reg_k]['R']])

                norm = np.sum(perc) if np.sum(perc) != 0 else 1.0
                perc = perc / norm

                if travelers > 0:
                    traveller_status = [int(p * travelers) for p in perc]  # np.random.choice(3, travelers, p=p)
                    for i in range(3):
                        s = label[i]
                        travel_to = np.random.randint(low=0, high=len(self.regions))

                        self.regions[travel_to][s] += traveller_status[i]
                        self.regions[reg_k][s] -= traveller_status[i]

                        # move to the infectious time queue of travel_to
                        if s == 'I':
                            if 'infection_time' not in self.regions[travel_to].keys():
                                self.regions[travel_to]['infection_time'] = self.regions[reg_k]['infection_time'][
                                                                            :traveller_status[i]]
                            else:
                                self.regions[travel_to]['infection_time'] = np.concatenate(
                                    [self.regions[travel_to]['infection_time'],
                                     self.regions[reg_k]['infection_time'][:traveller_status[i]]])

                            self.regions[reg_k]['infection_time'] = self.regions[reg_k]['infection_time'][
                                                                    traveller_status[i]:]

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

        size_isolated = sum([reg_v['isolated'] * reg_v['size'] for reg_v in self.regions.values()])

        uci = np.array([reg_v['uci'] for reg_v in self.regions.values()])
        infected = np.array([reg_v['I'] for reg_v in self.regions.values()])

        mortality = np.mean(infected / uci)  # a function of remaining UCI

        if sum([reg_v['I'] for reg_v in self.regions.values()]) == 0:
            terminal = 1
            r = 100

        r -= (mortality + np.mean(infected) + size_isolated)

        return r, terminal

    def action(self, action):
        self.hygiene = action[0]
        self.radius = action[1]

        for i in range(3):
            self.regions[i]['isolation'] = action[i + 2]

        for i in range(0, 4 * 3, 3):
            self.regions[i]['policy'] = action[i + 5:i + 8]

    def fill_queue(self, agent, queue):
        self.step()

        total_S = 0
        total_I = 0
        total_R = 0

        while total_R == 0 or total_I != 0:
            total_S = 0
            total_I = 0
            total_R = 0

            state = self.get_state()
            reward, terminal = self.get_reward_terminal()

            actions = agent(state)  # self.hygiene (0), self.radius, isolation (for region), p (region)
            # state self.S, self.I, self.R for regions

            self.step()

            next_state = self.get_state()

            for i, reg_v in enumerate(self.regions.values()):
                total_S += reg_v['S']
                total_I += reg_v['I']
                total_R += reg_v['R']

    def simulate(self, agent):
        total_S = 0
        total_I = 0
        total_R = 0

        regions_resumed = dict()

        # init
        for i, reg_v in enumerate(self.regions.values()):
            regions_resumed[i] = {
                'name': reg_v['name'],
                'size': reg_v['size'],
                'S': [reg_v['S']],
                'I': [reg_v['I']],
                'R': [0],
                'pos': {
                    'x': [],
                    'y': [],
                    'state': []
                }
            }

        while total_R < 50 or total_I > 0:
            total_S = 0
            total_I = 0
            total_R = 0

            state = self.get_state()
            reward, terminal = self.get_reward_terminal()

            actions = agent(state)  # self.hygiene (0), self.radius, isolation (for region), p (region)
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

                regions_resumed[i]['pos']['x'].append(reg_v['pos'][:, 0].tolist())
                regions_resumed[i]['pos']['y'].append(reg_v['pos'][:, 1].tolist())
                regions_resumed[i]['pos']['state'].append(reg_v['pos'][:, 2].tolist())

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


def model(population_cat,hygiene_cat,icu_cat,iratio_cat,size_cat):
    population = [population_cat,150,150]
    names = ['CAT', 'MAD', 'AND']
    sizes = [size_cat, 50, 45]
    
    UCI = [icu_cat, 10, 9]

    UCI = [9, 10, 9]

    sim = Simulator(population, names, sizes, UCI, 0, iratio_cat, hygiene_cat)
    sim.step()
    json_str = sim.simulate(agent)
    return json_str

def get_points(X_Cat,Y_Cat,Susceptible_cat,Infected_cat):
    Susceptible_cat_X=[]
    Infected_cat_X=[]
    Recovered_cat_X=[]
    Susceptible_cat_Y=[]
    Infected_cat_Y=[]
    Recovered_cat_Y=[]
    for i in range(len(X_Cat)-1):
        infected_start=Susceptible_cat[i+1]
        recovered_start=Susceptible_cat[i+1]+Infected_cat[i+1]
        Susceptible_cat_X.append(X_Cat[i+1][:infected_start])
        Infected_cat_X.append(X_Cat[i+1][infected_start:recovered_start])
        Recovered_cat_X.append(X_Cat[i+1][recovered_start:])
        Susceptible_cat_Y.append(Y_Cat[i+1][:infected_start])
        Infected_cat_Y.append(Y_Cat[i+1][infected_start:recovered_start])
        Recovered_cat_Y.append(Y_Cat[i+1][recovered_start:])
    return Susceptible_cat_X,Infected_cat_X,Recovered_cat_X,Susceptible_cat_Y,Infected_cat_Y,Recovered_cat_Y 

   


def get_points(X_Cat, Y_Cat, Susceptible_cat, Infected_cat):
    Susceptible_cat_X = []
    Infected_cat_X = []
    Recovered_cat_X = []
    Susceptible_cat_Y = []
    Infected_cat_Y = []
    Recovered_cat_Y = []
    for i in range(len(X_Cat) - 1):
        infected_start = Susceptible_cat[i + 1]
        recovered_start = Susceptible_cat[i + 1] + Infected_cat[i + 1]
        Susceptible_cat_X.append(X_Cat[i + 1][:infected_start])
        Infected_cat_X.append(X_Cat[i + 1][infected_start:recovered_start])
        Recovered_cat_X.append(X_Cat[i + 1][recovered_start:])
        Susceptible_cat_Y.append(Y_Cat[i + 1][:infected_start])
        Infected_cat_Y.append(Y_Cat[i + 1][infected_start:recovered_start])
        Recovered_cat_Y.append(Y_Cat[i + 1][recovered_start:])
    return Susceptible_cat_X, Infected_cat_X, Recovered_cat_X, Susceptible_cat_Y, Infected_cat_Y, Recovered_cat_Y


@app.route("/", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        population_cat= int(request.form["population"])
        hygiene_cat = float(request.form["hygiene"])
        icu_cat=int(request.form["icu"])
        iratio_cat=float(request.form["iRatio"])
        size_cat=int(request.form['size'])
        json_data=model(population_cat,hygiene_cat,icu_cat,iratio_cat,size_cat)
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

        Susceptible_cat=json_data[0]['S']
        Infected_cat=json_data[0]['I']
        Susceptible_and=json_data[1]['S']
        Infected_and=json_data[1]['I']
        Susceptible_val=json_data[2]['S']
        Infected_val=json_data[2]['I']
        X_Cat=json_data[0]['pos']['x']
        Y_Cat=json_data[0]['pos']['y']
        X_And=json_data[1]['pos']['x']
        Y_And=json_data[1]['pos']['y']
        X_Val=json_data[2]['pos']['x']
        Y_Val=json_data[2]['pos']['y']
        Susceptible_cat_X,Infected_cat_X,Recovered_cat_X,Susceptible_cat_Y,Infected_cat_Y,Recovered_cat_Y=get_points(X_Cat,Y_Cat,Susceptible_cat,Infected_cat)
        Susceptible_and_X,Infected_and_X,Recovered_and_X,Susceptible_and_Y,Infected_and_Y,Recovered_and_Y=get_points(X_And,Y_And,Susceptible_and,Infected_and)
        Susceptible_val_X,Infected_val_X,Recovered_val_X,Susceptible_val_Y,Infected_val_Y,Recovered_val_Y=get_points(X_Val,Y_Val,Susceptible_val,Infected_val)





        return render_template('index.html',data_aux=json_data,susceptible=Susceptible,infected=Infected,recovered=Recovered,x_pos=X_Cat,susceptiblecatx=Susceptible_cat_X,infectedcatx=Infected_cat_X,recoveredcatx=Recovered_cat_X,susceptiblecaty=Susceptible_cat_Y,infectedcaty=Infected_cat_Y,recoveredcaty=Recovered_cat_Y,susceptibleandx=Susceptible_and_X,infectedandx=Infected_and_X,recoveredandx=Recovered_and_X,susceptibleandy=Susceptible_and_Y,infectedandy=Infected_and_Y,recoveredandy=Recovered_and_Y,susceptiblevalx=Susceptible_val_X,infectedvalx=Infected_val_X,recoveredvalx=Recovered_val_X,susceptiblevaly=Susceptible_val_Y,infectedvaly=Infected_val_Y,recoveredvaly=Recovered_val_Y)

        return render_template('index.html',data_aux=json_data,susceptible=Susceptible,infected=Infected,recovered=Recovered,x_pos=X_Cat,susceptiblecatx=Susceptible_cat_X,infectedcatx=Infected_cat_X,recoveredcatx=Recovered_cat_X,susceptiblecaty=Susceptible_cat_Y,infectedcaty=Infected_cat_Y,recoveredcaty=Recovered_cat_Y)
    else:
        return render_template("inputs.html")

if __name__ == "__main__":
    app.run(debug=False)