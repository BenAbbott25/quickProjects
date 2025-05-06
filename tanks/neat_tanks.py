import matplotlib.pyplot as plt
import neat
import pygame
import numpy as np
from tanks import Game, Tank
from tqdm import tqdm
import pickle

# Define the fitness function
def eval_genomes(genomes, config=None):

    # Shuffle genomes to mix species
    np.random.shuffle(genomes)

    # Split genomes into batches of 10
    genome_batches = [genomes[i:i + 10] for i in range(0, len(genomes), 10)]

    for batch in tqdm(genome_batches):
        if batch == genome_batches[0]:
            watch_game = True
        else:
            watch_game = False
        game = Game(watch_game)
        for genome_id, genome in batch:
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            position = (np.random.randint(0, 800), np.random.randint(0, 600))
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            game.tanks[genome_id] = Tank(game, genome_id, position, color)

        while game.running:
            for genome_id, genome in batch:
                if genome_id not in game.tanks:
                    continue
                genome_tank = game.tanks[genome_id]
                inputs = [genome_tank.position[0] / 800, genome_tank.position[1] / 600, genome_tank.body_angle / (2 * np.pi), genome_tank.turret_angle / (2 * np.pi)]
                # Include inputs for all tanks in the batch
                for tank_id, tank in game.tanks.items():
                    if tank_id != genome_id:
                        inputs.extend([
                            tank.position[0] / 800,  # Normalize position
                            tank.position[1] / 600,
                            tank.body_angle / (2 * np.pi),
                            tank.turret_angle / (2 * np.pi),
                        ])
                
                # ensure 40 inputs
                while len(inputs) < 40:
                    inputs.append(0)

                # Get the outputs from the neural network
                outputs = net.activate(inputs)
                game.tanks[genome_id].update(outputs)

            game.update()

            if not game.running:
                for genome_id, genome in batch:
                    genome.fitness = game.fitnesses[genome_id]

# Load the NEAT configuration
def run_neat(config_file):
    pygame.init()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Load or create the population
    try:
        with open('population.pkl', 'rb') as f:
            population = pickle.load(f)
    except FileNotFoundError:
        population = neat.Population(config)

    # Add a reporter to show progress in the terminal
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_title('Sum of Fitnesses Over Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Sum of Fitnesses')
    line, = ax.plot([], [], 'r-')  # Initialize an empty line

    # Run the NEAT algorithm
    def update_plot():
        generation_fitness = [g.fitness for g in stats.most_fit_genomes]
        line.set_xdata(range(len(generation_fitness)))
        line.set_ydata(generation_fitness)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    for _ in range(100):
        population.run(eval_genomes, 1)
        update_plot()

    plt.ioff()  # Turn off interactive mode
    plt.show()

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(population.best_genome))

    # Save the population
    with open('population.pkl', 'wb') as f:
        pickle.dump(population, f)

if __name__ == "__main__":
    # Set the path to the NEAT configuration file
    config_path = "neat-config.txt"
    run_neat(config_path)