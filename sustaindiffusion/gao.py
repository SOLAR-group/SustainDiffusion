import calendar
import copy
import os
import random
import time
from PIL import Image
import numpy
import torch
from deap import algorithms, base, creator, tools
from diffusers import (
    StableDiffusion3Pipeline,
)
from huggingface_hub import login
from transformers import BlipProcessor, BlipForQuestionAnswering
from compel import Compel, ReturnedEmbeddingsType
from bias import eval_image_bias
from hf_token import hf_token
from img_quality import addBoxesImage, img2text
from objectives import Objectives, Fitness
import shutil
import json
import csv
from codecarbon import OfflineEmissionsTracker
import pandas as pd

print('!!! START OF FILE')
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
login(hf_token)

print('!!! HERE 1')
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    text_encoder_3=None,
    tokenizer_3=None,
)
print('!!! HERE 2')

# pipe = StableDiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2", torch_dtype=torch.float16
#     ).to(device)

pipe.to(device)
# pipe.enable_xformers_memory_efficient_attention()
if device == "cuda":
    pipe.enable_sequential_cpu_offload()

print('!!! PIPE CREATED, BEFORE PROMPT SELECTION')

# COMPEL for embedding generation
compel = Compel(
  tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
  text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  truncate_long_prompts=True,
  requires_pooled=[True, True],
  device=device
)

print("COMPEL CREATED")

# Functions to change prompt
def int_to_binary_and_select_elements(integer, element_list):
    binary_representation = bin(integer)[2:]
    selected_elements = []
    for i, digit in enumerate(binary_representation):
        if digit == "1":
            selected_elements.append(element_list[i])
    return selected_elements


def createNegativePrompt(selection, pos_prompt, plus_number):
    items = [
        "illustration",
        "painting",
        "drawing",
        "art",
        "sketch",
        "lowres",
        "error",
        "cropped",
        "worst quality",
        "low quality",
        "jpeg artifacts",
        "out of frame",
        "watermark",
        "signature",
    ]

    male_leading_prompts = [
        "ambitious",
        "assertive",
        "confident",
        "decisive",
        "determined",
        "intelligent",
        "outspoken",
        "self-confident",
        "stubborn",
        "unreasonable",
        "committed",
    ]

    # female_leading_prompts = [
    #     "supportive",
    #     "sensitive",
    #     "emotional",
    #     "gentle",
    #     "honest",
    #     "modest",
    #     "compassionate",
    #     "considerate",
    #     "pleasant",
    # ]
    # integer_input =  random.randint(0,2**len(fixed_length_list)-1)

    if selection[0] > 2 ** len(items) - 1:
        selection[0] %= 2 ** len(items) - 1
    selected_elements = int_to_binary_and_select_elements(selection[0], items)

    if selection[1] > 2 ** len(male_leading_prompts) - 1:
        selection[1] %= 2 ** len(male_leading_prompts) - 1

    # prompt weighting for male leading prompts
    # plus_number = random.randint(0, 4)

    prompt_weighting = ""
    for n in range(0, plus_number):
        prompt_weighting = prompt_weighting + "+"

    male_words = int_to_binary_and_select_elements(selection[1], male_leading_prompts)

    for i in range(0, len(male_words)):
        if male_words[i] in pos_prompt:
            male_words.remove(male_words[i])
        male_words[i] = male_words[i] + prompt_weighting
    
    for i in range(0, len(selected_elements)):
        selected_elements[i] = selected_elements[i] + prompt_weighting

    selected_elements += male_words
    # + ", " + ", ".join(male_words)

    # if selection[2] > 2 ** len(female_leading_prompts) - 1:
    #     selection[2] %= 2 ** len(female_leading_prompts) - 1
    # female_words = int_to_binary_and_select_elements(selection[2], items)
    # for word in female_words:
    #     if word in pos_prompt:
    #         female_words.remove(word)

    # selected_elements += female_words
    # + ", " + ", ".join(female_words)
    return ", ".join(selected_elements)


def createPosPrompt(prompt, selection, plus_number):
    items = [
        "photograph",
        "digital",
        "color",
        "Ultra Real",
        "film grain",
        "Kodak portra 800",
        "Depth of field 100mm",
        "overlapping compositions",
        "blended visuals",
        "trending on artstation",
        "award winning",
    ]

    # male_leading_prompts = [
    #     "ambitious",
    #     "assertive",
    #     "confident",
    #     "decisive",
    #     "determined",
    #     "intelligent",
    #     "outspoken",
    #     "self-confident",
    #     "stubborn",
    #     "unreasonable",
    #     "committed",
    # ]

    female_leading_prompts = [
        "supportive",
        "sensitive",
        "emotional",
        "gentle",
        "honest",
        "modest",
        "compassionate",
        "considerate",
        "pleasant",
    ]

    # For image quality
    # integer_input =  random.randint(0,2**len(fixed_length_list)-1)
    if selection[0] > 2 ** len(items) - 1:
        selection[0] %= 2 ** len(items) - 1
    selected_elements = int_to_binary_and_select_elements(selection[0], items)

    # if selection[1] > 2 ** len(male_leading_prompts) - 1:
    #     selection[1] %= 2 ** len(male_leading_prompts) - 1

    # selected_elements += int_to_binary_and_select_elements(
    #     selection[1], male_leading_prompts
    # )

    # + ", "
    # + ", ".join(
    #     int_to_binary_and_select_elements(selection[1], male_leading_prompts)
    # )

    # for female leading words
    if selection[2] > 2 ** len(female_leading_prompts) - 1:
        selection[2] %= 2 ** len(female_leading_prompts) - 1

    female_leading_selection = int_to_binary_and_select_elements(
        selection[2], female_leading_prompts
    )

    # plus_number = random.randint(0, 4)
    prompt_weighting = ""
    for n in range(0, plus_number):
        prompt_weighting = prompt_weighting + "+"

    for i in range(0, len(female_leading_selection)):

        female_leading_selection[i] = female_leading_selection[i] + prompt_weighting
    
    for i in range(len(selected_elements)):
        selected_elements[i] = selected_elements[i] + prompt_weighting

    selected_elements += female_leading_selection

    # + ","
    # + ",".join(
    #     int_to_binary_and_select_elements(selection[2], female_leading_prompts)
    # )

    return prompt + ", " + ", ".join(selected_elements)


# Function to generate images with Stable Diffusion
def generate_image(img_num, img_path, prompt, hyperparameters, folder_name, no_prompt):
    torch.cuda.empty_cache()
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    print("Generating image")
    print(hyperparameters)  # individual
    #  get hyperparameters
    denoising_steps = hyperparameters["denoising_steps"]
    guidance_scale = hyperparameters["guidance_scale"]

    if no_prompt:

        pos_prompt = createPosPrompt(
            prompt, hyperparameters["positive_prompt"], hyperparameters["weight"]
        )
        neg_prompt = createNegativePrompt(
            hyperparameters["negative_prompt"], pos_prompt, hyperparameters["weight"]
        )

        print("Positive Prompt: ", pos_prompt)
        print("Negative Prompt: ", neg_prompt)

        # Build prompt embeddings
        with torch.no_grad():
            embeds, pool = compel(pos_prompt)
            neg_embs, neg_pooled = compel.build_conditioning_tensor(neg_prompt)
            [embeds, neg_embs] = compel.pad_conditioning_tensors_to_same_length(conditionings=[embeds, neg_embs])
            embeds = torch.cat([embeds, embeds], -1)
            neg_embs = torch.cat([neg_embs, neg_embs], -1)

    # embeds = embeds.to(device='cpu', dtype=torch.float16)
    # pool = pool.to(device='cpu', dtype=torch.float16)
    # neg_embs = neg_embs.to(device='cpu', dtype=torch.float16)
    # neg_pooled = neg_pooled.to(device='cpu', dtype=torch.float16)

    # Generate the image
    final_path = os.path.join(img_path, prompt.replace(" ", "_"))
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    os.makedirs(final_path, exist_ok=True)

    image_save = os.path.join(
        folder_name,
        "all_imgs",
        f"d{hyperparameters['denoising_steps']}_g{hyperparameters['guidance_scale']}",
    )
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(image_save, exist_ok=True)

    print(f"Prompt: {prompt}")
    # print(f"Prompt: {prompt}, Positive: {pos_prompt}, Negative: {neg_prompt}")

    image_names = []
    # Generate Images
    energy_metrics = pd.DataFrame()
    for i in range(img_num):
        answer = ''
        timestamp = calendar.timegm(time.gmtime())
        output_file = f'output_{timestamp}.csv'
        tracker = OfflineEmissionsTracker(country_iso_code='GBR', output_file=output_file)
        while answer!= 'yes':
            print(f"Generating image no. {i} for individual: {hyperparameters}")

            with torch.no_grad():
                if no_prompt:
                    tracker.start()
                    image = pipe(
                        prompt=prompt.replace("\n", "").replace("$", "").replace("'", ""),
                        # negative_prompt=neg_prompt.replace("\n", "").replace("$", "").replace("'", ""),
                        guidance_scale=guidance_scale,
                        num_inference_steps=denoising_steps,
                        width = 512,
                        height = 512
                    ).images[0]
                    tracker.stop()
                else:
                    tracker.start()
                    image = pipe(
                        # prompt=prompt.replace("\n", "").replace("$", "").replace("'", ""),
                        # negative_prompt=neg_prompt.replace("\n", "").replace("$", "").replace("'", ""),
                        prompt_embeds=embeds,
                        pooled_prompt_embeds=pool,
                        negative_prompt_embeds=neg_embs,
                        negative_pooled_prompt_embeds=neg_pooled,
                        guidance_scale=guidance_scale,
                        num_inference_steps=denoising_steps,
                        width = 512,
                        height = 512
                    ).images[0]
                    tracker.stop()

            # Save the image
            image_name = f"image_{i}_{str(timestamp)}.png"
            image_path = os.path.join(final_path, image_name)
            image.save(image_path)

            # Check Image is human
            if os.path.isfile(image_path):
                raw_image = Image.open(image_path).convert('RGB')
                question = "Is this image of a human?"

                inputs = processor(raw_image, question, return_tensors="pt")
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)
                print(f"Is the Image human?: {answer}")
                # Delete non-human image
                if answer == 'no':
                    os.remove(image_path)
                    print(f"Image {image_name} was not human and has been deleted.")
        
        # SAVE ALL GENERATED IMAGES
        # image.save(os.path.join(image_save, image_name))
        image_names.append(image_name)
        emissions = pd.read_csv(output_file)
        emissions = emissions.tail(n=1)
        energy_metrics = pd.concat([energy_metrics, emissions[['cpu_energy', 'gpu_energy', 'duration']]], ignore_index=True)
        os.remove(output_file)

    print("Finish generating images")
    return final_path, image_names, energy_metrics

print('!!! JUST BEFORE GAO')
class GAOptimizer:
    """
    hyperparameters to tune:
    1.) Inference steps 0-100
    2.) Guidance scale 2-20?
    3.) Seed 0-2^9
    4.) Positive prompts (bias and image quality)
    5.) Negative prompts (bias and image quality)
        The same seed and prompt combo give the same exact image

    """

    def __init__(self, attributes={}) -> None:
        self.number_of_generations = int(attributes["number_of_generations"])
        self.mutation_probability = float(attributes["mutation_probability"])
        self.inner_mutation_probability = float(
            attributes["inner_mutation_probability"]
        )
        self.population_size = int(attributes["population_size"])
        #self.selection_size = int(attributes["selection_size"])
        self.crossover_probabiliy = int(attributes["crossover_probabiliy"])
        self.img_num = attributes["img_num"]
        self.mu = attributes["mu"]
        self.lambda_ = attributes["lambda"]
        self.prompt = attributes["prompt"]
        self.objective = attributes["objective"]
        self.folder_name = attributes["folder_name"]
        self.round = attributes["round"]
        self.fitness = attributes["fitness"]
        self.no_prompt = attributes['no_prompt']
        self.setup_deap()

    def create_individual(self):
        init_population = creator.Individual(
            {
                "denoising_steps": random.randint(30, 50),
                "guidance_scale": random.randint(1,20),
                "positive_prompt": [
                    random.randint(0, 2**11),
                    random.randint(0, 2**11),
                    random.randint(0, 2**8),
                ],
                "negative_prompt": [
                    random.randint(0, 2**14),
                    random.randint(0, 2**10),
                    random.randint(0, 2**8),
                ],
                "weight": random.randint(0, 5),
            }
        )
        return init_population

    # helper functions so I can use built in crossover method

    def individual_to_list(self, individual_dict):
        print('!!! IN GAO')
        return [
            individual_dict["denoising_steps"],
            individual_dict["guidance_scale"],
            individual_dict["positive_prompt"],
            individual_dict["negative_prompt"],
            individual_dict["weight"]
        ]

    def list_to_individual(self, individual_list):
        keys = [
            "denoising_steps",
            "guidance_scale",
            "positive_prompt",
            "negative_prompt",
            "weight"
        ]
        return dict(zip(keys, individual_list))
    
    def individual_to_hashable(self, individual):
        # Convert the list values in the individual dictionary to tuples
        return (
            individual["denoising_steps"],
            individual["guidance_scale"],
            tuple(individual["positive_prompt"]),  # Convert list to tuple
            tuple(individual["negative_prompt"])   # Convert list to tuple
        )
    

    def unique_population(self, n):
        """Create a population of unique individuals."""
        seen = set()
        population = []
        attempts = 0
        max_attempts = n * 10  # Prevent infinite loops

        while len(population) < n and attempts < max_attempts:
            # Create a new individual with the DEAP Individual class
            individual = self.create_individual()

            # Get hashable representation for uniqueness check
            hashable_ind = self.individual_to_hashable(individual)

            if hashable_ind not in seen:
                seen.add(hashable_ind)
                population.append(individual)  # Add the Individual object directly
            attempts += 1
        
        return population
    
    # after image generation
    def eval_fitness(self, individual):
        print(f"Generating Images for: \\n {individual}")

        # generate images
        image_folder, image_names, energy_metrics = generate_image(
            self.img_num,
            f"experiment_images_{self.fitness}_{self.round}",
            self.prompt,
            individual,
            self.folder_name,
            self.no_prompt
        )


        # skin_tone_counts = {"light": 0, "dark": 0}
        gender_counts = {"male": 0, "female": 0}
        ethnicity_counts = {"asian": 0, "white": 0, "black": 0, "arab": 0}

        avgPrecision = 0
        totalCount = 0

        print("Analysing images...")
        print("Total count:", str(totalCount))
        for img_name in image_names:
            image_path = os.path.join(image_folder, img_name)
            # Fitness function for fairness
            # eval_image_bias(image_path, skin_tone_counts, gender_counts)
            gender_counts, ethnicity_counts = eval_image_bias(
                image_path, gender_counts, ethnicity_counts, processor, model
            )
            # Eval image quality
            counting, boxesInfo = img2text(image_path)
            print(counting)
            addBoxesImage(image_path, boxesInfo)
            print(boxesInfo)
            for box in boxesInfo:
                totalCount += 1
                avgPrecision += box[2]

        print("Calculating Fitness ...")

        if avgPrecision == 0:
            image_quality = 0
        else:
            image_quality = avgPrecision / totalCount

        # Compute bias fitness
 
        # gender fitness
        female_ratio = gender_counts["female"] / self.img_num
        male_ratio = gender_counts["male"] / self.img_num

        gender_fitness = abs(female_ratio - 0.5) + abs(male_ratio - 0.5)

        # ethnicity fitness
        total_ethnic_count = (
            ethnicity_counts["white"]
            + ethnicity_counts["asian"]
            + ethnicity_counts["black"]
            + ethnicity_counts["arab"]
        )
        white_ratio = ethnicity_counts["white"] / total_ethnic_count
        asian_ratio = ethnicity_counts["asian"] / total_ethnic_count
        arab_ratio = ethnicity_counts["arab"] / total_ethnic_count
        black_ratio = ethnicity_counts["black"] / total_ethnic_count

        ethnicity_fitness = abs(max(white_ratio, asian_ratio, arab_ratio, black_ratio) -
            min(white_ratio, asian_ratio, arab_ratio, black_ratio) )

        # energy metrics
        cpu_energy = energy_metrics['cpu_energy'].median()
        gpu_energy = energy_metrics['gpu_energy'].median()
        duration = energy_metrics['duration'].median()

        print(
            # f"Individual: {individual} \\n Skintone Fitness: {skin_tone_fitness} \\n  Gender Fitness: {gender_fitness} \\nCombined Fitness: {combined_fitness}"
            f"Individual: {individual} \n  Gender Fitness: {gender_fitness} \n Gender counts: {gender_counts} \n Ethnicity Fitness: {ethnicity_fitness} \n Ethnicity counts: {ethnicity_counts} \n cpu {cpu_energy} \n gpu {gpu_energy} \n duration {duration}"
        )
        # For saving fitness values to csv files
        fieldnames = [
            "Individual",
            "Image Quality",
            "Gender Fitness",
            "Ethnicity Fitness",
            "Gender Count",
            "Ethnicity Count",
            "CPU Energy",
            "GPU Energy",
            "Duration"
        ]

        # row = [json.dumps(individual), image_quality, gender_fitness, ethnicity_fitness, json.dumps(gender_counts), json.dumps(ethnicity_counts)]
        row_dict = {
            "Individual": json.dumps(individual),
            "Image Quality": image_quality,
            "Gender Fitness": gender_fitness,
            "Ethnicity Fitness": ethnicity_fitness,
            "Gender Count": json.dumps(gender_counts),
            "Ethnicity Count": json.dumps(ethnicity_counts),
            "CPU Energy": json.dumps(cpu_energy),
            "GPU Energy": json.dumps(gpu_energy),
            "Duration": json.dumps(duration)
        }

        csv_file = f"{self.folder_name}/fitness.csv"
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header if the file is empty
            if file.tell() == 0:
                writer.writeheader()

            # Write the row
            writer.writerow(row_dict)

        print("Image quality:", str(image_quality))
        # individual['image_quality'] = image_quality
        # individual['gender_bias'] = gender_fitness
        # individual['ethnicity_bias'] = ethnicity_fitness
        # individual['cpu_energy'] = cpu_energy
        # individual['gpu_energy'] = gpu_energy
        # individual['duration'] = duration
  
        if self.objective == Objectives.FAIRNESS.value:
            if self.fitness == Fitness.WEIGHT.value:
                return image_quality - gender_fitness - ethnicity_fitness,
            else:
                return (image_quality, gender_fitness, ethnicity_fitness)
        if self.objective == Objectives.ENERGY.value:
            if self.fitness == Fitness.WEIGHT.value:
                return image_quality - cpu_energy - gpu_energy - duration,
            else:
                return (image_quality, cpu_energy, gpu_energy, duration)
        if self.objective == Objectives.CPU.value:
            return (image_quality, cpu_energy)
        if self.objective == Objectives.GPU.value:
            return (image_quality, gpu_energy)
        if self.objective == Objectives.DURATION.value:
            return (image_quality, duration)
        if self.fitness == Fitness.WEIGHT.value:
            return image_quality - gender_fitness - ethnicity_fitness - cpu_energy - gpu_energy - duration,
        if self.objective == Objectives.ALL.value:
            return (image_quality, gender_fitness, ethnicity_fitness, cpu_energy)
        if self.objective == Objectives.IMAGE.value:
            return image_quality,
        return (image_quality, gender_fitness, ethnicity_fitness, cpu_energy, gpu_energy, duration)

    def crossover(self, individual1, individual2):
        print("Crossing over ...")
        # randomly select which genes to crossover using DEAP library

        # convert to lists
        list_ind1 = self.individual_to_list(individual1)
        list_ind2 = self.individual_to_list(individual2)

        tools.cxUniform(list_ind1, list_ind2, indpb=0.5)

        # convert back to dictionaries
        new_ind1 = self.list_to_individual(list_ind1)
        new_ind2 = self.list_to_individual(list_ind2)

        return new_ind1, new_ind2

    def mutate(self, individual):
        print("Mutating ...")
        ind = copy.copy(individual)
        new_ind = self.create_individual()

        for chromsome in individual.keys():
            if random.random() < self.inner_mutation_probability:
                ind[chromsome] = new_ind[chromsome]
        return (ind,)


    def setup_deap(self):
        if self.fitness == Fitness.WEIGHT.value or self.objective == Objectives.IMAGE.value:
            creator.create("FitnessMin", base.Fitness, weights=(1,))
        else:
            if self.objective == Objectives.CPU.value or self.objective == Objectives.GPU.value or self.objective == Objectives.DURATION.value:
                creator.create("FitnessMin", base.Fitness, weights=(1, -1))
            elif self.objective == Objectives.ALL.value:
                creator.create("FitnessMin", base.Fitness, weights=(1, -1, -1, -1))
            else:
                creator.create("FitnessMin", base.Fitness, weights=(1, -1, -1, -1, -1, -1))
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.create_individual
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self.eval_fitness)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        if self.fitness == Fitness.NSGAIII.value:
            ref_points = tools.uniform_reference_points(nobj=3, p=6)
            self.toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
        elif self.fitness == Fitness.WEIGHT.value or self.objective == Objectives.IMAGE:
            self.toolbox.register("select", tools.selTournamentDCD, 5)
        else:
            self.toolbox.register("select", tools.selNSGA2)
        print("Optimising ", self.objective)

    def optimization(self):
        # collect statistsics for the individuals in population
        hof = tools.ParetoFront()
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        population = self.toolbox.population(n=self.population_size)

        print("Running GAO ...")
        # run simple GA. offspring = thefinal population after GA is finished.
        _, logbook = algorithms.eaMuCommaLambda(
            population,
            self.toolbox,
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb=self.crossover_probabiliy,
            mutpb=self.mutation_probability,
            ngen=self.number_of_generations,
            stats=stats,
            halloffame=hof
        )

        # select the best individual
        best = tools.selBest(population, k=1)
        # pareto_front = tools.sortNondominated(
        #     population, len(population), first_front_only=True
        # )
        return best[0], hof, logbook
