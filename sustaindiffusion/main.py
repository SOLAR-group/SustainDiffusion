from gao import GAOptimizer
from argparse import ArgumentParser
from objectives import Objectives, Fitness
import pandas as pd
import csv
import torch


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--num_gen", type=int)
    parser.add_argument("--pop_size", type=int)
    parser.add_argument("--objective")
    parser.add_argument("--imgs_to_generate", type=int, default=20)
    parser.add_argument("--round", type=int)
    parser.add_argument('--fitness', type=str, choices=[
        Fitness.NSGAII.value,
        Fitness.NSGAIII.value,
        Fitness.WEIGHT.value
    ], default=Fitness.NSGAII.value)
    parser.add_argument('--no_prompt', action='store_true')

    args = parser.parse_args()
    folder_name = f"results_{args.objective}/results/round_{args.round}"
    
    print("Starting the experiment")
    prompt = "Photo portrait of a Software Engineer that codes"
    #prompt = "Photo portrait of a person that codes"
    attributes = {
        "number_of_generations": args.num_gen,
        "mutation_probability": 0.2,
        "inner_mutation_probability": 0.2,
        "population_size": args.pop_size,
        "selection_size": 5,
        "crossover_probabiliy": 0.8,
        "objective": args.objective,
        "img_num": args.imgs_to_generate,
        "mu": 5,
        "lambda": 5,
        "prompt": prompt,
        "round": args.round,
        "folder_name": folder_name,
        "fitness": args.fitness,
        "no_prompt": args.no_prompt
    }

    ga = GAOptimizer(attributes)
    print(f"Beginning GAO round {args.round}")
    # To record time taken to reach optimum
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    starter.record()
    best_individual, pareto_front, logbook = ga.optimization()
    ender.record()

    print(f"Best Individual:, {best_individual}")
    print(f"Best Fitness:, {best_individual.fitness.values}")
    print(f"Offspring:, {pareto_front}")
    print(f"Logbook:, {logbook}")
    print(f"Time taken: {starter.elapsed_time(ender)/ 1000}")


    # Write the lobgook to a CSV file
    logdf = pd.DataFrame(logbook)
    logdf.to_csv(f"{folder_name}/logbook.csv")

    with open(f"{folder_name}/results.csv", "a", newline="") as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        # Write the dictionary string to the CSV file
        writer.writerow([ind for ind in pareto_front])
        writer.writerow([ind.fitness.values for ind in pareto_front])

    print("Done")
