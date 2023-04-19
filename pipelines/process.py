import torch
import io

# set root directory
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model import run_worflow, filter_bboxes_from_outputs, plot_finetuned_results
from PIL import Image
from datetime import datetime
from expiration_date_process import run_expiration_date_workflow

from firebase_admin import credentials, initialize_app, storage, db, firestore


import matplotlib.pyplot as plt
import cv2
from matplotlib.figure import Figure
from transformers import pipeline

import requests


products_images = {
    'apple': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Fapple.png?alt=media&token=ec72ba85-b52b-4a99-8c50-8f410f6e6d2d',
    'banana': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Fbanana.png?alt=media&token=c0ff9c28-4cca-411b-b303-b96b5ab09306',
    'brocoli': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Fbrocoli.png?alt=media&token=587d7cd2-560d-4c88-bd18-b8d5d6dc376d',
    'butter': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Fbutter.png?alt=media&token=5b9dd218-febd-48ea-9987-5488193b6ca6',
    'carrot': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Fcarrot.png?alt=media&token=89046cb8-a4f8-4c55-bce9-43d13047756f',
    'cottage': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Fcottage.png?alt=media&token=26e66ff7-d87d-43d2-90ac-2475d6bd4e94',
    'cream': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2FsweetCream.png?alt=media&token=cb95c430-59e1-428e-91e6-f4150349dd0b',
    'milk': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Fmilk.png?alt=media&token=eac29753-b984-48c6-9552-2b5f765be50b',
    'mustart': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Fmustard.png?alt=media&token=28f91402-efc1-4f12-b5cd-99f9dc8fdeb1',
    'oragne': 'https://firebasestorage.googleapis.com/v0/b/fridgeit-d17ae.appspot.com/o/products%2Forange.png?alt=media&token=6abac4ee-7c83-4487-aadf-3088caa9d18c',
}


# Our fridgeIT dataset classes
finetuned_classes = [
    'butter', 'cottage', 'milk', 'mustard', 'cream', 'banana', 'apple', 'orange', 'broccoli', 'carrot'
]

coco_classes = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


cred = credentials.Certificate("./FBserviceAccountKey.json")
initialize_app(cred, {'storageBucket': 'fridgeit-d17ae.appspot.com'})
db = firestore.client()
user_document = db.collection(
        'EPaDIxTXxINXm88w7xoBPRaNcFh1').document('user_data')


# file_name = 'current.png'
cropped_folder_path = 'cropped/'
history_cropped_path = 'cropped_history/'
source = "current_picture/current.png"
dest = './uploads/'
local_history_cropped = './history_cropped/'


def delete_cropped_images_from_storage():
    # current dateTime
    now = datetime.now()

    # convert to string
    date_time_str = now.strftime("%Y-%m-%d#%H:%M:%S")
    count = 0
    
    # delete local history folder
    for file in os.listdir(dest):
        os.remove(dest + file)

    # delete local uploads folder
    for file in os.listdir(local_history_cropped):
        os.remove(local_history_cropped + file)
    # delete cropped images from firebase storage
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=cropped_folder_path)
    # download the last two croppen picture from the cropped folder in firebase storage
    for blob in blobs:
        # for every blob, download the blob picture
        file = local_history_cropped+'{}#{}.png'.format(date_time_str, count)
        blob.download_to_filename(file)

        # save the picture in the history_cropped folder in firebase storage
        history_blob = bucket.blob(
            history_cropped_path + '{}#{}.png'.format(date_time_str, count))
        history_blob.upload_from_filename(file)

        # delete the picture from the cropped folder in firebase storage
        blob.delete()

        count += 1

def delete_old_detected_products():
    # Delete the product in the recently detected product firebase firestore array
    user_document.update({'recently_detected_products': firestore.DELETE_FIELD})
    

def load_model():
    detr_model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    
    finetune_num_classes = 5
    finetune_detr_model = torch.hub.load('facebookresearch/detr',
                           'detr_resnet50',
                           pretrained=False,
                           num_classes=finetune_num_classes)

    # Loading checkpoint
    checkpoint = torch.load('checkpoint.pth', map_location='cpu')

    finetune_detr_model.load_state_dict(checkpoint["model"], strict=False)

    return detr_model.eval(), finetune_detr_model.eval()


def get_image_from_storage():
    bucket = storage.bucket()
    blob = bucket.blob(source)
    file = dest + 'current.png'
    blob.download_to_filename(file)
    img = Image.open(file)
    return img


def crop(image_obj, box, saved_location):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    cropped_image = image_obj.crop((x1, y1, x2, y2))
    cropped_image.save(saved_location)
    cropped_image.show()


def crop_and_store_products(img, finetune_boxes, finetune_probs, detr_boxes, detr_probs, captioner):    
        
    for ind, box in enumerate(finetune_boxes):
        prob = finetune_probs[ind]
        cl = prob.argmax()
        
        class_name = finetuned_classes[cl]
        score = f'{prob[cl]:0.2f}'
        cropped_image, product_obj = save_cropped_image(img,class_name, score, box)
        # Get the expiration date of the product with jinhybr/OCR-Donut-CORD
        data = captioner(cropped_image)
        expiration_date_results = run_expiration_date_workflow(
            data[0]['generated_text'])
        
        product_obj['expiration_date'] = expiration_date_results
        
        update_user_document(product_obj)
        
        
    for ind, box in enumerate(detr_boxes):
        prob = detr_probs[ind]
        cl = prob.argmax()
        
        if coco_classes[cl] not in finetuned_classes:
            continue
        class_name = coco_classes[cl]
        score = f'{prob[cl]:0.2f}'
        cropped_image, product_obj = save_cropped_image(img,class_name, score, box)
        update_user_document(product_obj)

        
# Save cropped images to firebase storage
def save_cropped_image(img, class_name, score, box):
        
    # current dateTime
    now = datetime.now()
    # convert to string
    date_time_str = now.strftime("%Y-%m-%d#%H:%M:%S")

    crop_path = '{}.png'.format(f'{class_name}#{score}')
    saved_location = ('./uploads/{}'.format(crop_path))  
    
    crop(img, box, saved_location)
       
    bucket = storage.bucket()
    blob = bucket.blob('cropped/{}'.format(crop_path))
    blob.upload_from_filename('./uploads/{}'.format(crop_path))
    
    im = Image.open('./uploads/{}'.format(crop_path))
    product_obj = {'name': class_name.capitalize(), 'image': products_images[class_name],
                    'created_date': date_time_str, 'quantity': 1, 'score': f'{score}'} 
    return img, product_obj
   
        
def update_user_document(product_obj):
    doc_ref = user_document.get()
    # If the product has expiration date
    if "recently_detected_products" in doc_ref.to_dict() :
        recently_detected_products = doc_ref.get('recently_detected_products')     
        all_detected_products = doc_ref.get('all_detected_products')  
        # update recent_detected_products array       
        if 'expiration_date' in product_obj:
            for product in recently_detected_products:
                if product['expiration_date'] == 'not founded':
                    continue
                if(product['name'].lower() == product_obj['name'] and 
                   product['expiration_date'] == product_obj['expiration_date'] and
                   product['score'] == product_obj['score']):
                    product['quantity'] += 1
                    product['created_date'] = product_obj['created_date']
                    user_document.update({'recently_detected_products': recently_detected_products})
                    break
            else:
                user_document.update({'recently_detected_products': firestore.ArrayUnion([product_obj])})

        else:               
            for product in recently_detected_products:
                    if(product['name'] == product_obj['name'] and product['score'] == product_obj['score']):
                        product['quantity'] += 1
                        product['created_date'] = product_obj['created_date']
                        user_document.update({'recently_detected_products': recently_detected_products})
                        break
            else:
                user_document.update({'recently_detected_products': firestore.ArrayUnion([product_obj])})
        # update all_detected_products array 
        del product_obj['quantity']  
        user_document.update({"all_detected_products": firestore.ArrayUnion([product_obj])})

    # If the product has not expiration date
    else:
        user_document.update({'recently_detected_products': [product_obj]}) 
        del product_obj['quantity']   
        user_document.update({"all_detected_products": [product_obj]})



def main():

    captioner = pipeline('image-to-text', model='jinhybr/OCR-Donut-CORD')

    # delete all images from storage in folder cropped and related document
    delete_cropped_images_from_storage()
    
    delete_old_detected_products()

    detr, finetune = load_model()
    
    # load image from storage current_picture/current.png
    im = get_image_from_storage()
    # im=Image.open('apple.png')


    # predict model on image
    detr_outputs, finetune_outputs = run_worflow(im, detr, finetune)
    
    # getting bboxes and probs
    finetune_probs, finetune_boxes = filter_bboxes_from_outputs(finetune_outputs, im=im)
    detr_probs, detr_boxes = filter_bboxes_from_outputs(detr_outputs, im=im)
    
    # cropping finetuned products
    crop_and_store_products(im, finetune_boxes, finetune_probs, detr_boxes, detr_probs, captioner)
    
    # cropping finetuned products
    #crop_and_store_coco_classes(im, detr_boxes, detr_prob)
    
    return plot_finetuned_results(im, finetune_probs, finetune_boxes, detr_probs, detr_boxes)


if __name__ == '__main__':
    main()
