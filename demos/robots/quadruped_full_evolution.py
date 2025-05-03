import math
import numpy as np
import pyrosim
import random
import matplotlib.pyplot as plt



# Hyperparameters
POP_SIZE = 30
GENERATIONS = 100
MUTATION_RATE = 0.1

# Parameter ranges
PHASE_RANGE = (0, 2 * math.pi)
FREQ_RANGE = (0.5, 2.0)
AMP_RANGE = (0.2, 1.0)

# Random parameters
HEIGHT = 0.3
EPS = 0.05
#np.random.seed(0)

def random_individual():
    return np.array([
        np.random.uniform(*PHASE_RANGE),
        np.random.uniform(*FREQ_RANGE),
        np.random.uniform(*AMP_RANGE)
    ] * 8)  # 8 joints

def mutate(individual):
    mutant = individual.copy()
    for i in range(len(mutant)):
        if np.random.rand() < MUTATION_RATE:
            if i % 3 == 0:
                mutant[i] = np.random.uniform(*PHASE_RANGE)
            elif i % 3 == 1:
                mutant[i] = np.random.uniform(*FREQ_RANGE)
            else:
                mutant[i] = np.random.uniform(*AMP_RANGE)
    return mutant

def cross_over(parent1, parent2):
    crossover_point = np.random.randint(0, len(parent1))
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def crossover_uniform(parent1, parent2):
    return [random.choice([gene1, gene2]) for gene1, gene2 in zip(parent1, parent2)]


def send_to_simulator(sim, individual, eval_time):
    main_body = sim.send_box(x=0, y=0, z=HEIGHT+EPS,
                             length=HEIGHT, width=HEIGHT,
                             height=EPS*2.0, mass=1)
    pos_sensor_id = sim.send_position_sensor(body_id=main_body)

    thighs = [0]*4
    shins = [0]*4
    hips = [0]*4
    knees = [0]*4
    motor_neurons = []

    delta = math.pi / 2.0

    for i in range(4):
        theta = delta * i
        x_pos = math.cos(theta) * HEIGHT
        y_pos = math.sin(theta) * HEIGHT

        thighs[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=HEIGHT+EPS,
                                      r1=x_pos, r2=y_pos, r3=0,
                                      length=HEIGHT, radius=EPS, capped=True)

        hips[i] = sim.send_hinge_joint(main_body, thighs[i],
                                       x=x_pos/2.0, y=y_pos/2.0, z=HEIGHT+EPS,
                                       n1=-y_pos, n2=x_pos, n3=0,
                                       lo=-math.pi/4.0, hi=math.pi/4.0,
                                       speed=1.0)

        x_pos2 = math.cos(theta)*1.5*HEIGHT
        y_pos2 = math.sin(theta)*1.5*HEIGHT

        shins[i] = sim.send_cylinder(x=x_pos2, y=y_pos2, z=(HEIGHT+EPS)/2.0,
                                     r1=0, r2=0, r3=1,
                                     length=HEIGHT, radius=EPS,
                                     mass=1., capped=True)

        knees[i] = sim.send_hinge_joint(thighs[i], shins[i],
                                        x=x_pos2, y=y_pos2, z=HEIGHT+EPS,
                                        n1=-y_pos, n2=x_pos, n3=0,
                                        lo=-math.pi/4.0, hi=math.pi/4.0)

        # Voeg motorneuronen toe voor heup en knie
        motor_neurons.append(hips[i])
        motor_neurons.append(knees[i])

    # Add sine wave oscillattions to motor neurons
    for idx, joint in enumerate(motor_neurons):
        phase = individual[idx * 3]
        freq = individual[idx * 3 + 1]
        amp = individual[idx * 3 + 2]
        def oscillation(t, phase=phase, freq=freq, amp=amp):
            return amp * math.sin(2 * math.pi * freq * t + phase)
        values = [oscillation(i * sim.dt) for i in range(eval_time)]
        neuron_id = sim.send_user_input_neuron(values)
        sim.send_synapse(source_neuron_id=neuron_id,
                         target_neuron_id=sim.send_motor_neuron(joint_id=joint),
                         weight=1.0)

    sim.create_collision_matrix('all')

    return main_body, pos_sensor_id

def evaluate(individual, seconds = 30.0, dt = 0.05, play_blind = True):
    eval_time = int(seconds / dt)
    gravity = -1.0

    sim = pyrosim.Simulator(eval_time=eval_time, debug=False,
                            play_blind=play_blind, gravity=gravity, dt=dt)

    main_body, pos_sensor_id = send_to_simulator(sim, individual, eval_time)
    sim.start()
    sim.wait_to_finish()
    

    pos_data_x = sim.get_sensor_data(sensor_id=pos_sensor_id, svi=0)
    pos_data_y = sim.get_sensor_data(sensor_id=pos_sensor_id, svi=1)
    final_y = pos_data_y[-1]  # laatste timestep, y-component
    final_x = pos_data_x[-1]  # laatste timestep, x-component
    return final_x**2 + final_y**2  # fitness is de afstand van de oorsprong



TOP_K = 5  # number of top individuals to retain and mutate from

if __name__ == "__main__":
    population = [random_individual() for _ in range(POP_SIZE)]
    highest_fitness = 0
    best_individual = None

    best_fitnesses = []
    avg_fitnesses = []

    for generation in range(GENERATIONS):
        fitnesses = [evaluate(ind) for ind in population]
        sorted_indices = np.argsort(fitnesses)[::-1]
        best_idx = sorted_indices[0]
        best = population[best_idx]

        best_fitness = fitnesses[best_idx]
        avg_fitness = np.mean(fitnesses)

        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)

        print(f"Gen {generation}: Best fitness = {best_fitness:.2f}, Avg fitness = {avg_fitness:.2f}")

        if best_fitness > highest_fitness:
            if best_fitness - highest_fitness > 10 or generation == 0:
                print("Found a significantly better individual!")
                evaluate(best, seconds=10, dt=0.05, play_blind=False)

            highest_fitness = best_fitness
            best_individual = best


        elites = [population[i] for i in sorted_indices[:TOP_K]]
        population = [mutate(random.choice(elites)) for _ in range(POP_SIZE)]

        # add crossover between elites
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = random.sample(elites, 2)
            child = crossover_uniform(parent1, parent2)
            population.append(child)

    print(f"Showing best individual with fitness {highest_fitness:.2f}")
    evaluate(best_individual, seconds=30, dt=0.05, play_blind=False)

    plt.figure(figsize=(10, 6))
    plt.plot(best_fitnesses, label="Best Fitness")
    plt.plot(avg_fitnesses, label="Average Fitness", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

