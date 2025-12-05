import calendar
import os
import random
import time
from PIL import Image
import torch
from diffusers import (
    StableDiffusion3Pipeline,
)
from huggingface_hub import login
from transformers import BlipProcessor, BlipForQuestionAnswering
from compel import Compel, ReturnedEmbeddingsType
from hf_token import hf_token
from img_quality import addBoxesImage, img2text
import shutil
import json
import csv
from codecarbon import OfflineEmissionsTracker
import pandas as pd
from transformers import  BlipForQuestionAnswering
import cv2
from PIL import Image
from pandas import *
import ast

# SETUP
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
login(hf_token)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    text_encoder_3=None,
    tokenizer_3=None,
)

pipe.to(device)
if device == "cuda":
    pipe.enable_sequential_cpu_offload()

compel = Compel(
  tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
  text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  truncate_long_prompts=True,
  requires_pooled=[True, True],
  device=device
)

# Relevant files
ga_results_file = 's3_nsgaii_results_all/results.csv'
prompts_list = 's3_prompt_comparison/se_prompts_new.txt'

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
    if selection[0] > 2 ** len(items) - 1:
        selection[0] %= 2 ** len(items) - 1
    selected_elements = int_to_binary_and_select_elements(selection[0], items)


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


    return prompt + ", " + ", ".join(selected_elements)

def get_gender_with_blip(img_path, blip_processor, blip_model):
    # image_path = os.path.join(path, f"image_{i}.png")
    img = cv2.imread(img_path)
    question = "What is the gender of the person in the image?"
    inputs = blip_processor(images=img, text=question, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    answer = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return answer


def get_ethnicity_with_blip(img_path, blip_processor, blip_model):
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    # model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    answer = ""
    if os.path.isfile(img_path):

        raw_image = Image.open(img_path).convert("RGB")
        question = "What ethnicity is this person?"
        inputs = blip_processor(raw_image, question, return_tensors="pt")

        out = blip_model.generate(**inputs)
        answer = blip_processor.decode(out[0], skip_special_tokens=True)

        # Asian, Black, White, Arab
        if (
            str(answer).lower() == "caucasian"
            or str(answer).lower() == "white"
            or str(answer).lower() == "italian"
            or str(answer).lower() == "german"
            or str(answer).lower() == "hispanic"
        ):
            answer = "white"
        elif str(answer).lower() == "indian" or str(answer).lower() == "asian":
            answer = "asian"
        elif (
            str(answer).lower() == "black"
            or str(answer).lower() == "african american"
            or str(answer).lower() == "african"
        ):
            answer = "black"
        elif str(answer).lower() == "arab" or str(answer).lower() == "middle eastern":
            answer = "arab"
        else:
            answer = "other"

    return answer


def eval_image_bias(image_path, gender_counts, ethnicity_counts, blip_processor, blip_model):
    ethnicity = get_ethnicity_with_blip(image_path, blip_processor, blip_model)
    print("ethnicity: ", ethnicity)
    gender = get_gender_with_blip(image_path, blip_processor, blip_model)
    print("gender: ", gender)

    if ethnicity == "other":
        ethnicity_counts["white"] += 1
        ethnicity_counts["asian"] += 1
        ethnicity_counts["black"] += 1
        ethnicity_counts["arab"] += 1
    else:
        ethnicity_counts[ethnicity] += 1

    if gender == "female":
        gender_counts["female"] += 1
    elif gender == "male":
        gender_counts["male"] += 1

    return gender_counts, ethnicity_counts


# Function to generate images with default parameters
def generate_image_baseline(img_num, prompt, folder_name):
    torch.cuda.empty_cache()
    final_path = os.path.join(folder_name, 'Images')
    os.makedirs(final_path, exist_ok=True)
    print("Generating image")


    # Generate the image
    print(f"Prompt: {prompt}")

    image_names = []

    # Generate Images
    energy_metrics = pd.DataFrame()
    for i in range(img_num):
        answer = ''
        timestamp = calendar.timegm(time.gmtime())
        output_file = f'output_{timestamp}.csv'
        tracker = OfflineEmissionsTracker(country_iso_code='GBR', output_file=output_file)
        while answer!= 'yes':
            print(f"Generating image no. {i}")
            tracker.start()
            image = pipe(prompt=prompt).images[0]
            tracker.stop()

            # Save the image
            image_name = f"image_{i}.png"
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
        image_names.append(image_name)
        emissions = pd.read_csv(output_file)
        emissions = emissions.tail(n=1)
        energy_metrics = pd.concat([energy_metrics, emissions[['cpu_energy', 'gpu_energy', 'duration']]], ignore_index=True)
        os.remove(output_file)

    print("Finish generating images")
    return final_path, image_names, energy_metrics
    
# Function to generate images with Stable Diffusion with custom hypeparameters
def generate_image(img_num, prompt, hyperparameters, folder_name):
    torch.cuda.empty_cache()
    final_path = os.path.join(folder_name, 'Images')
    os.makedirs(final_path, exist_ok=True)
    # os.makedirs(img_path, exist_ok=True)
    print("Generating image")
    print(hyperparameters)  # individual
    #  get hyperparameters
    denoising_steps = hyperparameters["denoising_steps"]
    guidance_scale = hyperparameters["guidance_scale"]
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


    # Generate the image
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
                tracker.start()
                image = pipe(
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
            image_name = f"image_{i}.png"
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
        image_names.append(image_name)
        emissions = pd.read_csv(output_file)
        emissions = emissions.tail(n=1)
        energy_metrics = pd.concat([energy_metrics, emissions[['cpu_energy', 'gpu_energy', 'duration']]], ignore_index=True)
        os.remove(output_file)

    print("Finish generating images")
    return final_path, image_names, energy_metrics


def create_individual():
    init_population = {
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
    
    return init_population



def individual_to_hashable(individual):
    # Convert the list values in the individual dictionary to tuples
    return (
        individual["denoising_steps"],
        individual["guidance_scale"],
        tuple(individual["positive_prompt"]),  # Convert list to tuple
        tuple(individual["negative_prompt"])   # Convert list to tuple
    )


def unique_population(n):
    """Create a population of unique individuals."""
    seen = set()
    population = []
    attempts = 0
    max_attempts = n * 10  # Prevent infinite loops

    while len(population) < n and attempts < max_attempts:
        # Create a new individual with the DEAP Individual class
        individual = create_individual()

        # Get hashable representation for uniqueness check
        hashable_ind = individual_to_hashable(individual)

        if hashable_ind not in seen:
            seen.add(hashable_ind)
            population.append(individual)  # Add the Individual object directly
        attempts += 1
    
    return population



# after image generation
def eval_fitness(individual, folder_name, prompt, baseline = None):
    print(f"Generating Images for: \\n {individual}")

    # generate images
    if baseline == None:
        image_folder, image_names, energy_metrics = generate_image(
            10,
            prompt,
            individual,
            folder_name
        )
    else:
        image_folder, image_names, energy_metrics = generate_image_baseline(
        10,
        prompt,
        folder_name
    )


    gender_counts = {"male": 0, "female": 0}
    ethnicity_counts = {"asian": 0, "white": 0, "black": 0, "arab": 0}

    avgPrecision = 0
    totalCount = 0

    print("Analysing images...")
    print("Total count:", str(totalCount))
    for img_name in image_names:
        image_path = os.path.join(image_folder, img_name)
        # Fitness function for fairness
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
    female_ratio = gender_counts["female"] / 10
    male_ratio = gender_counts["male"] / 10

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

    csv_file = f"{folder_name}/fitness.csv"
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        # Write the row
        writer.writerow(row_dict)

    print("Image quality:", str(image_quality))


    return (image_quality, gender_fitness, ethnicity_fitness, cpu_energy, gpu_energy, duration)


def parse_text_to_dict(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing text: {e}")
        return None
    
prompts = open(prompts_list, 'r')
data = read_csv(ga_results_file)
individual_list = data['Solution'].tolist()

for line in prompts:
    prompt = line.strip().replace('_', ' ').replace('\n','').replace('$', '')
    # img_name = line.strip().replace('\n','').replace('$', '')
    folder_ga = os.path.join('s3_prompt_comparison', prompt, 'GA Individual')
    os.makedirs(folder_ga, exist_ok=True)

    # Random selected individual from pareto front
    # hyperparameter = parse_text_to_dict(random.choice(individual_list))
    # while hyperparameter == "\n" or hyperparameter == None or hyperparameter["denoising_steps"] == None:
    #     hyperparameter = parse_text_to_dict(random.choice(individual_list))
    # print("Running Randomly selected from GA...")
    # eval_fitness(hyperparameter, folder_ga, prompt)

    # # Random Search
    # folder_random = os.path.join('s3_prompt_comparison', prompt, 'Random Search')
    # os.makedirs(folder_random, exist_ok=True)
    # individual = create_individual()
    # print("Running Random Search...")
    # eval_fitness(individual, folder_random, prompt)

    # # Baseline
    # folder_baseline = os.path.join('s3_prompt_comparison', prompt, 'Baseline')
    # os.makedirs(folder_baseline, exist_ok=True)
    # print("Running Baseline...")
    # eval_fitness(individual, folder_baseline, prompt, 1)

    # Fair Prompt

    folder_fair = os.path.join('s3_prompt_comparison', prompt, 'Fair')
    os.makedirs(folder_fair, exist_ok=True)
    prompt = prompt + ", such that it fairly represents different genders and ethnicites"
    eval_fitness(None, folder_fair, prompt, 'Fair')
    