import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from detr_model.model import run_worflow, filter_bboxes_from_outputs, plot_finetuned_results
from utils.firebase_config import user_document, cropped_folder_storage_path, history_cropped_storage_path, bucket
from expiration_date_process import run_expiration_date_workflow
from utils.classes_list import coco_classes, finetuned_classes
from utils.products_image import products_image_list

# set root directory
from firebase_admin import storage, firestore
from transformers import pipeline
from datetime import datetime
from PIL import Image
import torch

""" 
We provide four images from the current_picture in our firbase storage.
This images include an objects from our classes.

You can use the following names of images as a value for source variable :
    current1.jpg , current2.jpeg, current3.jpeg, current4.jpeg
"""
source = "current_picture/current1.jpeg"
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
    blobs = bucket.list_blobs(prefix=cropped_folder_storage_path)
    # download the last two croppen picture from the cropped folder in firebase storage
    for blob in blobs:
        # for every blob, download the blob picture
        file = local_history_cropped+'{}#{}.png'.format(date_time_str, count)
        blob.download_to_filename(file)

        # save the picture in the history_cropped folder in firebase storage
        history_blob = bucket.blob(
            history_cropped_storage_path + '{}#{}.png'.format(date_time_str, count))
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
    
    state_dict = torch.load('./detr_model/model.pt')
    finetune_detr_model = torch.hub.load('facebookresearch/detr',
                        'detr_resnet50',
                        pretrained=False,
                        num_classes=finetune_num_classes)
    finetune_detr_model.load_state_dict(state_dict)
    
    return detr_model.eval(), finetune_detr_model.eval()


def get_image_from_storage():
    blob = bucket.blob(source)
    file = dest + 'current.png'
    blob.download_to_filename(file)
    img = Image.open(file).convert('RGB')
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
        current_date_and_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # current dateTime
        prob = finetune_probs[ind]
        cl = prob.argmax()
        
        class_name = finetuned_classes[cl]
        score = f'{prob[cl]:0.2f}'
        cropped_image, product_obj = save_cropped_image(img, class_name, score, box, current_date_and_time)
        
        # Get the expiration date of the product with jinhybr/OCR-Donut-CORD
        data = captioner(cropped_image)
        expiration_date_results = run_expiration_date_workflow(
            data[0]['generated_text'])
        
        product_obj['expiration_date'] = expiration_date_results
        
        update_user_document(product_obj)
        
        
    for ind, box in enumerate(detr_boxes):
        current_date_and_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prob = detr_probs[ind]
        cl = prob.argmax()
        
        if coco_classes[cl] not in finetuned_classes:
            continue
        class_name = coco_classes[cl]
        score = f'{prob[cl]:0.2f}'
        cropped_image, product_obj = save_cropped_image(img,class_name, score, box, current_date_and_time)
        update_user_document(product_obj)

        
# Save cropped images to firebase storage
def save_cropped_image(img, class_name, score, box, current_date_and_time):

    crop_path = '{}.png'.format(f'{current_date_and_time}#{class_name}#{score}')
    saved_location = ('./uploads/{}'.format(crop_path))  
    
    crop(img, box, saved_location)
       
    blob = bucket.blob('cropped/{}'.format(crop_path))
    blob.upload_from_filename('./uploads/{}'.format(crop_path))
    
    im = Image.open('./uploads/{}'.format(crop_path))
    product_obj = {'name': class_name.capitalize(), 'image': products_image_list[class_name],
                    'created_date': current_date_and_time, 'score': f'{score}'} 
    return im, product_obj
   
        
def update_user_document(product_obj):

    user_document.update({'recently_detected_products': firestore.ArrayUnion([product_obj])})  
    user_document.update({"all_detected_products": firestore.ArrayUnion([product_obj])})



def main():
    
    # load detr model and OCR-Donut-CORD model
    captioner = pipeline('image-to-text', model='jinhybr/OCR-Donut-CORD')

    # delete all images from storage in folder cropped and related document
    delete_cropped_images_from_storage()
    
    delete_old_detected_products()

    # load detr model and finetue_detr model
    detr, finetune = load_model()
    
    # load image from storage current_picture/current.png
    im = get_image_from_storage()


    # predict model on image
    detr_outputs, finetune_outputs = run_worflow(im, detr, finetune)
    
    # get bboxes and probs
    finetune_probs, finetune_boxes = filter_bboxes_from_outputs(finetune_outputs, im=im)
    detr_probs, detr_boxes = filter_bboxes_from_outputs(detr_outputs, im=im)
    
    # crop and store products
    crop_and_store_products(im, finetune_boxes, finetune_probs, detr_boxes, detr_probs, captioner)
    
    
    return plot_finetuned_results(im, finetune_probs, finetune_boxes, detr_probs, detr_boxes)


if __name__ == '__main__':
    main()

