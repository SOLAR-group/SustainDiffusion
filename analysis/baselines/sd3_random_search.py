import csv
import pandas as pd

import calendar
import os
import random
import time
from PIL import Image
from argparse import ArgumentParser
import torch
from codecarbon import OfflineEmissionsTracker
from diffusers import (
    StableDiffusion3Pipeline,
)
from huggingface_hub import login
from transformers import BlipProcessor, BlipForQuestionAnswering
from bias import eval_image_bias
from hf_token import hf_token
from img_quality import addBoxesImage, img2text
from objectives import Objectives
from codecarbon import OfflineEmissionsTracker
import shutil
import json
import csv
from compel import Compel, ReturnedEmbeddingsType

parser = ArgumentParser()
parser.add_argument("--round")

args = parser.parse_args()
# !! need to make a master file of all the combined logboks for the GAO for SD3!!
logbook = 'sd3_random_search/logbook_master.csv'
df = pd.read_csv(logbook)

# runs is average of all evals in logbook
runs = int((df['nevals'].mean())) +1
print(f"number of runs is: {runs}")
print(df.head(), df.tail())
print(f"Average of nevals is: {df['nevals'].mean()}")
device = "cuda" if torch.cuda.is_available() else "mps"
output_file = f"sd3_random_search/stats/emissions_round_{args.round}.csv"
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

login(hf_token)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    text_encoder_3=None,
    tokenizer_3=None,
)

# pipe = StableDiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2", torch_dtype=torch.float16
#     ).to(device)

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

folder = os.path.join("sd3_random_search/Images", f"Round_{args.round}" )
os.makedirs(folder, exist_ok=True)

# Functions to change prompt
def int_to_binary_and_select_elements(integer, element_list):
    binary_representation = bin(integer)[2:]
    selected_elements = []
    for i, digit in enumerate(binary_representation):
        if digit == "1":
            selected_elements.append(element_list[i])
    return selected_elements


def createNegativePrompt(selection, pos_prompt 
                         ,plus_number
                         ):
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
        prompt_weighting = prompt_weighting + "-"

    male_words = int_to_binary_and_select_elements(selection[1], male_leading_prompts)
    # for word in male_words:
    #     if word in pos_prompt:
    #         male_words.remove(word)
    # selected_elements += male_words

    for i in range(0, len(male_words)):
        if male_words[i] in pos_prompt:
            male_words.remove(male_words[i])
        male_words[i] = male_words[i] + prompt_weighting
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


def createPosPrompt(prompt, selection
                    , plus_number
                    ):
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

    selected_elements+=female_leading_selection

    # plus_number = random.randint(0, 4)

    prompt_weighting = ""
    for n in range(0, plus_number):
        prompt_weighting = prompt_weighting + "+"

    for i in range(0, len(female_leading_selection)):

        female_leading_selection[i] = female_leading_selection[i] + prompt_weighting

    selected_elements += female_leading_selection

    # + ","
    # + ",".join(
    #     int_to_binary_and_select_elements(selection[2], female_leading_prompts)
    # )

    return prompt + ", " + ", ".join(selected_elements)


# Function to generate images with Stable Diffusion
def generate_image(img_num, img_path, prompt, hyperparameters):
    torch.cuda.empty_cache()
    print("Generating image")
    print(hyperparameters)  # individual
    #  get hyperparameters
    denoising_steps = hyperparameters["denoising_steps"]
    guidance_scale = hyperparameters["guidance_scale"]
    pos_prompt = createPosPrompt(
        prompt, hyperparameters["positive_prompt"], 
        hyperparameters["weight"],
    )
    neg_prompt = createNegativePrompt(
        hyperparameters["negative_prompt"], pos_prompt, 
        hyperparameters["weight"],
    )

    print("Prompt: ", pos_prompt)
    print("Negative Prompt: ", neg_prompt)
    # Generate the image
    print(f"Prompt: {prompt}, Positive: {pos_prompt}, Negative: {neg_prompt}")

    with torch.no_grad():
        embeds, pool = compel(pos_prompt)
        neg_embs, neg_pooled = compel.build_conditioning_tensor(neg_prompt)
        [embeds, neg_embs] = compel.pad_conditioning_tensors_to_same_length(conditionings=[embeds, neg_embs])
        embeds = torch.cat([embeds, embeds], -1)
        neg_embs = torch.cat([neg_embs, neg_embs], -1)

    image_names = []
    os.mkdir(img_path)
    # Generate Images
    for i in range(img_num):
        answer = ''
        while answer!= 'yes':
            print(f"Generating image no. {i} for individual: {hyperparameters}")

            with torch.no_grad():
                tracker = OfflineEmissionsTracker(
                    country_iso_code="GBR", output_file=output_file
                )
                tracker.start()
                image = pipe(
                    # prompt=prompt.replace("\n", "").replace("$", "").replace("'", ""),
                    prompt_embeds=embeds,
                    pooled_prompt_embeds=pool,
                    negative_prompt_embeds=neg_embs,
                    negative_pooled_prompt_embeds=neg_pooled,
                    guidance_scale=guidance_scale,
                    num_inference_steps=denoising_steps,
                    # negative_prompt=neg_prompt,
                    # num_images_per_prompt=img_num,
                ).images[0]
                tracker.stop()

            # Save the image
            timestamp = calendar.timegm(time.gmtime())
            image_name = f"image_{i}_{str(timestamp)}.png"
            image_path = os.path.join(img_path, image_name)
            image.save(image_path)

            # Check Image is human
            if os.path.isfile(image_path):
                raw_image = Image.open(image_path).convert('RGB')
                question = "Is this image of a human?"

                inputs = processor(raw_image, question, return_tensors="pt")
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)

                # Delete non-human image
                if answer == 'no':
                    os.remove(image_path)
                    print(f"Image {image_name} was not human and has been deleted.")                

        image_names.append(image_name)

    print("Finish generating images")
    return image_names

def eval_fitness(individual):
    print(f"Generating Images for: \\n {individual}")
    image_folder = os.path.join(folder, f"d{individual['denoising_steps']}_g{individual['guidance_scale']}")

    # generate images
    image_names = generate_image(
        10,
        image_folder,
        "Photo portrait of a Software Engineer that codes",
        individual,
    )

    gender_counts = {"male": 0, "female": 0}
    ethnicity_counts = {"asian": 0, "white": 0, "black": 0, "arab": 0}

    avgPrecision = 0
    totalCount = 0

    print("Analysing images...")
    print("Total count:", str(totalCount))

    for img_name in image_names:
        image_path = os.path.join(image_folder, img_name)

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

    ethnicity_fitness = (
        abs(white_ratio - 0.25)
        + abs(asian_ratio - 0.25)
        + abs(arab_ratio - 0.25)
        + abs(black_ratio - 0.25)
    )
    print(
        # f"Individual: {individual} \\n Skintone Fitness: {skin_tone_fitness} \\n  Gender Fitness: {gender_fitness} \\nCombined Fitness: {combined_fitness}"
        f"Individual: {individual} \n  Gender Fitness: {gender_fitness} \n Gender counts: {gender_counts} \n Ethnicity Fitness: {ethnicity_fitness} \n Ethnicity counts: {ethnicity_counts} "
    )
    # For saving fitness values to csv files
    fieldnames = [
        "Individual",
        "Image Quality",
        "Gender Fitness",
        "Ethnicity Fitness",
        "Gender Count",
        "Ethnicity Count",
    ]

    # row = [json.dumps(individual), image_quality, gender_fitness, ethnicity_fitness, json.dumps(gender_counts), json.dumps(ethnicity_counts)]
    row_dict = {
        "Individual": json.dumps(individual),
        "Image Quality": image_quality,
        "Gender Fitness": gender_fitness,
        "Ethnicity Fitness": ethnicity_fitness,
        "Gender Count": json.dumps(gender_counts),
        "Ethnicity Count": json.dumps(ethnicity_counts),
    }

    csv_file = f"sd3_random_search/stats/fitness_round_{args.round}.csv"
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        # Write the row
        writer.writerow(row_dict)

    print("Image quality:", str(image_quality))

    return (image_quality, gender_fitness, ethnicity_fitness)

def create_individual():
    init_population = {
        "denoising_steps": random.randint(25, 50),
        "guidance_scale": random.randint(1, 20),
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
        "weight": random.randint(0, 10),
        'seed': random.randint(0,2**9)
    }
    
    return init_population

best_individual = None
best_fitness = None
best_fitness_score = 10

individuals_list = set()

for n in range(runs):
    print(f"Running round {n} of {runs}")

    # Get a random individual - avoid repeats
    individual = create_individual()
    scale_steps = (individual["denoising_steps"], individual["guidance_scale"])
    while scale_steps in individuals_list:
        individuals_list = create_individual()
        scale_steps = (individual["denoising_steps"], individual["guidance_scale"])
    individuals_list.add(scale_steps)
    
    # Generate 10 images for that individual and Evaluate fitness of individual
    result = eval_fitness(individual)
    quality, gender, ethnicity = result[0], result[1], result[2]

    # This is not needed here

    # Update best fitness and best individuals
    # new_fitness_score = sum(gender, ethnicity)
    # if  new_fitness_score < best_fitness_score:
    #     best_fitness = (quality, gender, ethnicity)
    #     best_fitness_score = new_fitness_score
    #     best_individual = individual

