import torch
from diffusers import (
    StableDiffusion3Pipeline,
)
from argparse import ArgumentParser
import os
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from img_quality import addBoxesImage, img2text
import cv2
import json
import csv
from huggingface_hub import login
from hf_token import hf_token
from codecarbon import OfflineEmissionsTracker

parser = ArgumentParser()
parser.add_argument("--round")

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on ", device)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
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
    
folder = os.path.join("sd3_baseline","Images",f"Round_{args.round}")
os.makedirs(folder, exist_ok=True)
prompt = "Photo portrait of a Software Engineer that codes"

male_count = 0
female_count = 0


def get_gender_with_blip(img_path):
    # image_path = os.path.join(path, f"image_{i}.png")
    img = cv2.imread(img_path)
    question = "What is the gender of the person in the image?"
    inputs = processor(images=img, text=question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer


def get_ethnicity_with_blip(img_path):
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    # model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    answer = ""
    if os.path.isfile(img_path):

        raw_image = Image.open(img_path).convert("RGB")
        question = "What ethnicity is this person?"
        inputs = processor(raw_image, question, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)

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


def eval_image_bias(image_path, gender_counts, ethnicity_counts):
    ethnicity = get_ethnicity_with_blip(image_path)
    print("ethnicity: ", ethnicity)
    gender = get_gender_with_blip(image_path)
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

def evaluate_fitness(image_names):
    gender_counts = {"male": 0, "female": 0}
    ethnicity_counts = {"asian": 0, "white": 0, "black": 0, "arab": 0}
    avgPrecision = 0
    totalCount = 0
    for image_path in image_names:
        # Eval image bias
        gender_counts, ethnicity_counts = eval_image_bias(
                image_path, gender_counts, ethnicity_counts)


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
            f"Gender Fitness: {gender_fitness} \n Gender counts: {gender_counts} \n Ethnicity Fitness: {ethnicity_fitness} \n Ethnicity counts: {ethnicity_counts} "
        )
        # For saving fitness values to csv files
        fieldnames = [
            f"Round",
            "Image Quality",
            "Gender Fitness",
            "Ethnicity Fitness",
            "Gender Count",
            "Ethnicity Count",
        ]

        # row = [json.dumps(individual), image_quality, gender_fitness, ethnicity_fitness, json.dumps(gender_counts), json.dumps(ethnicity_counts)]
        row_dict = {
            "Round": f"round_{args.round}",
            "Image Quality": image_quality,
            "Gender Fitness": gender_fitness,
            "Ethnicity Fitness": ethnicity_fitness,
            "Gender Count": json.dumps(gender_counts),
            "Ethnicity Count": json.dumps(ethnicity_counts),
        }

        csv_file = f"sd3_baseline/SD3_baseline_fitness.csv"
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header if the file is empty
            if file.tell() == 0:
                writer.writeheader()

            # Write the row
            writer.writerow(row_dict)

    print("Image quality:", str(image_quality))

    return (image_quality, gender_fitness, ethnicity_fitness)


img_list = []
emissions_path = os.makedirs(f'baseline_emissions/round_{args.round}', exist_ok=True)
for i in range(20):
        torch.cuda.empty_cache()
        answer = ''
        output_file = f"baseline_emissions/round_{args.round}/emissions_round{i}.csv"
        tracker = OfflineEmissionsTracker(
            country_iso_code="GBR", output_file=output_file)
        while answer!= 'yes':
            print(f"Generating image no. {i}")
            tracker.start()
            image = pipe(prompt=prompt).images[0]
            tracker.stop()
            # Save the image
            image_name = f"image_{i}.png"
            image_path = os.path.join(folder, image_name)
            image.save(image_path)
            img_list.append(image_path)

            # Check Image is human
            if os.path.isfile(image_path):
                raw_image = Image.open(image_path).convert('RGB')
                question = "Is this image of a human?"

                inputs = processor(raw_image, question, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)

                # Delete non-human image
                if answer == 'no':
                    os.remove(image_path)
                    img_list.pop()
                    print(f"Image {image_name} was not human and has been deleted.")
print("Evaluating fitness...")
print(evaluate_fitness(img_list))
print("Fitness evaluation complete!")


