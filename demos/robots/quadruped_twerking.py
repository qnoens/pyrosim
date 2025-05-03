import math
import numpy as np
import pyrosim
import random

HEIGHT = 0.3
EPS = 0.05
np.random.seed(0)

def send_to_simulator(sim):
    main_body = sim.send_box(x=0, y=0, z=HEIGHT+EPS,
                             length=HEIGHT, width=HEIGHT,
                             height=EPS*2.0, mass=1)

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

    for idx, joint in enumerate(motor_neurons):
        # generate random parameter values


        phase_offset = (idx % 2) * math.pi  # om en om in tegenfase
        frequency = 1.0  # Hz
        amplitude = 0.5  # schaal van de beweging

        # Definieer de functie voor de neuron
        def oscillation(t, phase=phase_offset, freq=frequency, amp=amplitude):
            return amp * math.sin(2 * math.pi * freq * t + phase)

        # Maak een lijst van waarden over de simulatieperiode
        time_steps = int(sim.eval_time)
        dt = sim.dt
        values = [oscillation(i * dt) for i in range(time_steps)]

        # Stuur de functie-neuron en verbind deze met de motor
        neuron_id = sim.send_user_input_neuron(values)
        sim.send_synapse(source_neuron_id=neuron_id,
                         target_neuron_id=sim.send_motor_neuron(joint_id=joint),
                         weight=1.0)

    sim.create_collision_matrix('all')

if __name__ == "__main__":
    seconds = 100.0
    dt = 0.05
    eval_time = int(seconds / dt)
    print(eval_time)
    gravity = -1.0

    sim = pyrosim.Simulator(eval_time=eval_time, debug=True,
                            play_paused=False,
                            gravity=gravity,
                            play_blind=False,
                            use_textures=True,
                            capture=False,
                            dt=dt)

    send_to_simulator(sim)
    sim.start()
    sim.wait_to_finish()