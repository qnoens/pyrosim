import math
import numpy as np
import math
import pyrosim

HEIGHT = 0.3
EPS = 0.05
np.random.seed(0)
BOX_MASS = 1000

def add_blocks(sim, level, box_size, step_height):
    height = level * step_height

    dist = level*box_size

    long_block_length = (2*level + 1)*box_size
    short_block_length = (2*level - 1)*box_size

    box = sim.send_box(
        x=0, y=dist, z=height/2,
        length=box_size, width=long_block_length, height=height,
        mass=BOX_MASS
    )
    sim.send_fixed_joint(box, pyrosim.Simulator.WORLD)
    box = sim.send_box(
        x=0, y=-dist, z=height/2,
        length=box_size, width=long_block_length, height=height,
        mass=BOX_MASS
    )
    sim.send_fixed_joint(box, pyrosim.Simulator.WORLD)
    box = sim.send_box(
        x=dist, y=0, z=height/2,
        length=short_block_length, width=box_size, height=height,
        mass=BOX_MASS
    )
    sim.send_fixed_joint(box, pyrosim.Simulator.WORLD)
    box = sim.send_box(
        x=-dist, y=0, z=height/2,
        length=short_block_length, width=box_size, height=height,
        mass=BOX_MASS
    )
    sim.send_fixed_joint(box, pyrosim.Simulator.WORLD)

def add_square_valley(sim, levels, box_size, step_height):
    for level in range(1, levels + 1):
        add_blocks(sim, level, box_size, step_height)

def send_to_simulator(sim, weight_matrix = None):
    add_square_valley(sim, 3, 1, 0.2)

    main_body = sim.send_box(x=0, y=0, z=HEIGHT+EPS,
                             length=HEIGHT, width=HEIGHT,
                             height=EPS*2.0, mass=1)

    # id arrays
    thighs = [0]*4
    shins = [0]*4
    hips = [0]*4
    knees = [0]*4
    foot_sensors = [0]*4
    sensor_neurons = [0]*5
    motor_neurons = [0]*8

    delta = float(math.pi)/2.0

    # quadruped is a box with one leg on each side
    # each leg consists thigh and shin cylinders
    # with hip and knee joints
    # each shin/foot then has a touch sensor
    for i in range(4):
        theta = delta*i
        x_pos = math.cos(theta)*HEIGHT
        y_pos = math.sin(theta)*HEIGHT

        thighs[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=HEIGHT+EPS,
                                      r1=x_pos, r2=y_pos, r3=0,
                                      length=HEIGHT, radius=EPS, capped=True
                                      )

        hips[i] = sim.send_hinge_joint(main_body, thighs[i],
                                       x=x_pos/2.0, y=y_pos/2.0, z=HEIGHT+EPS,
                                       n1=-y_pos, n2=x_pos, n3=0,
                                       lo=-math.pi/4.0, hi=math.pi/4.0,
                                       speed=1.0)

        motor_neurons[i] = sim.send_motor_neuron(joint_id=hips[i])

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

        motor_neurons[i+4] = sim.send_motor_neuron(knees[i])
        foot_sensors[i] = sim.send_touch_sensor(shins[i])
        sensor_neurons[i] = sim.send_sensor_neuron(foot_sensors[i])

    # count = 0
    # developing synapses linearly change from the start value to the
    # end value over the course of start time to end time
    # Here we connect each sensor to each motor, pulling weights from
    # the weight matrix
    # for source_id in sensor_neurons:
    #     for target_id in motor_neurons:
    #         count += 1
    #         start_weight = weight_matrix[source_id, target_id, 0]
    #         end_weight = weight_matrix[source_id, target_id, 1]
    #         start_time = weight_matrix[source_id, target_id, 2]
    #         end_time = weight_matrix[source_id, target_id, 3]
    #         sim.send_developing_synapse(source_id, target_id,
    #                                     start_weight=start_weight,
    #                                     end_weight=end_weight,
    #                                     start_time=start_time, 
    #                                     end_time=end_time)

    # layouts are useful for other things not relevant to this example
    # layout = {'thighs': thighs,
    #           'shins': shins,
    #           'hips': hips,
    #           'knees': knees,
    #           'feet': foot_sensors,
    #           'sensor_neurons': sensor_neurons,
    #           'motor_neurons': motor_neurons}

    # send the box towards the origin with specified force
    # proportional to its mass
    MASS = 3
    # sim.send_external_force(env_box, x=0, y=0, z=MASS*20., time=300)
    # sim.send_external_force(env_box, x=-MASS*70., y=MASS*73., z=0, time=320)

    sim.create_collision_matrix('all')

    # return layout

if __name__ == "__main__":

    seconds = 120.0
    dt = 0.05
    eval_time = int(seconds/dt)
    print(eval_time)
    gravity = -1.0

    sim = pyrosim.Simulator(xyz=[7,0,4], hpr = [180,-30,0], eval_time=eval_time, debug=False,
                               play_paused=False,
                               gravity=gravity,
                               play_blind=False,
                               use_textures=False,
                               capture=False,
                               dt=dt)
    num_sensors = 5
    num_motors = 8

    # our weight matrix specifies the values for the 
    # starting and ending weights as well as the starting and
    # ending times for the synapses
    # weight_matrix = np.random.rand(num_sensors+num_motors,
    #                                num_sensors+num_motors, 4)
    # weight_matrix[:, :, 0:1] = weight_matrix[:, :, 0:1]*2.-1.

    send_to_simulator(sim)
    sim.start()

    sim.wait_to_finish()
