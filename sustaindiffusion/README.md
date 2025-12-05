# Sustain Diffusion

This folder contains the source code of **SustainDiffusion**.

## Usage

- Make sure to install the required dependencies as described in the [README](../README.md).
- Run the following command to start the SustainDiffusion:

```bash
python main.py --num_gen <num_gen> --pop_size <pop_size> --objective <objective> --imgs_to_generate <imgs_to_generate> --round <round> --fitness <fitness> [--no_prompt]
```

Where:

- `<num_gen>`: Number of generations for the genetic algorithm.
- `<pop_size>`: Population size for the genetic algorithm.
- `<objective>`: The objective function to optimize. Must be one of:
  - `fairness`: optimise only for gender and ethnic bias.
  - `energy`: optimise for cpu, gpu and duration.
  - `cpu`: optimise for cpu usage.
  - `gpu`: optimise for gpu usage.
  - `duration`: optimise for duration.
  - `image`: optimise for image quality.
  - `all`: optimise for all the above.
- `<imgs_to_generate>`: Number of images to generate.
- `<round>`: The round number for the experiment.
- `<fitness>`: The search strategy to use.
- `--no_prompt`: Optional flag to disable prompts engineering.

Results will be saved in a folder named `results_<objective>/results/round_<round>`, where `<objective>` is the objective function and `<round>` is the round number.
