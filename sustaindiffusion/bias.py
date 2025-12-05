from transformers import AutoProcessor, BlipForQuestionAnswering
import cv2
from PIL import Image
# from derm_ita import get_ita, get_fitzpatrick_type
# from facenet_pytorch import MTCNN
import os
from transformers import BlipForQuestionAnswering


# Functions to evaluate gender and skin tone bias

# Load BLIP VQA for Gender detection
# blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
# blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


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


# mtcnn = MTCNN(keep_all=True)


# def crop_face(image):
#     boxes, _ = mtcnn.detect(image)

#     # no faces detected
#     if boxes is None:
#         return None

#     # crop the first face found
#     box = boxes[0]
#     cropped_face = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

#     return cropped_face


# def get_skin_tone_score(image_path):
#     # image_path = os.path.join(path, f"image_{i}.png")
#     img = cv2.imread(
#         image_path
#     )  # optinal if image is already generated, change argument to image_path instead

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )
#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     if faces is not None and len(faces) > 0:
#         x, y, w, h = faces[0]
#         cropped_face = img[y : y + h, x : x + w]

#         cropped_face_image = Image.fromarray(
#             cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
#         )
#         whole_image_ita = get_ita(cropped_face_image)
#         skin_type = get_fitzpatrick_type(whole_image_ita)

#         return skin_type
#     else:
#         return 0


# def eval_image_bias(image_path, skin_tone_counts, gender_counts):
#     skin_tone_score = get_skin_tone_score(image_path)
#     print("skintone score: ")
#     print(skin_tone_score)
#     gender = get_gender_with_blip(image_path)
#     print("gender: ", gender)
#     # for skintone we consider 1-3 to be light skin, and 4-6 to be darler skin
#     if skin_tone_score in [1, 2, 3]:
#         skin_tone_counts["light"] += 1
#     else:
#         skin_tone_counts["dark"] += 1

#     # for simplicity, ive grouped gender to discrete categories. More likely female = 0, more likely male = 1
#     if gender == "female":
#         gender_counts["female"] += 1
#     elif gender == "male":
#         gender_counts["male"] += 1


# JUST GENDER FOR NOW
