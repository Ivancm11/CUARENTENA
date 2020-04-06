from simulation import Simulator
from agent import Agent, ReplayBuffer
import numpy as np

if __name__ == '__main__':
    a = Agent()
    q = ReplayBuffer(100)

    #print(a.select_action([100, 100,100, 100, 11, 0, 2,3,4]))

    for i in range(100):
        population = np.random.randint(low=90, high=150, size=3)
        
        names = ['CAT', 'MAD', 'AND']
        sizes = [90, 100, 90]
        UCI = [9, 10, 9]

        print("Simulating")
        sim = Simulator(population, names, sizes, UCI, 0, 2, 0.2)
        total_reward = sim.fill_queue(lambda state: a.select_action(state, i), q)
        print("Sim Done")

        print("Mean Reward: ", total_reward)
        a.fit(q, 100, 0.9)
        print("Fit Done")

        if i%10 == 0:
            print("%i - Updating" % i)
            a.update_target()
    
    a.save()
