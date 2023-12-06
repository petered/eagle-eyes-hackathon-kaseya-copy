import json
from PIL import Image
import time
import os

# Assuming the JSON data is stored in a file named 'annotations.json'
with open('dataset.json') as f:
    data = json.load(f)
for item_key, item_value in data.items():
    print(f"Processing item: {item_key}")
    for image_index, image_info in enumerate(data[item_key]['images']):
    # Assuming you want to crop the bounding box from the first image of the 'WALK_DAY_ROCK' entry
    #image_info = data[item_key]['images'][1]
        source_path = image_info['source_path']
        if image_info['annotations'] is not None:
            for annotation_index, annotation in enumerate(image_info['annotations']):
                ijhw_box = annotation['ijhw_box']  # The bounding box
            
            #ijhw_box = image_info['annotations'][0]['ijhw_box']  # First annotation's bounding box

            # Load the image
                image = Image.open(source_path)

                # Crop the bounding box from the image
                # ijhw_box contains [i, j, h, w]
                top = ijhw_box[0] - 50
                left = ijhw_box[1] - 50
                

                # The crop method takes a box tuple in the form (left, upper, right, lower)
                cropped_image = image.crop((left, top, left + 100, top + 100))

                # Display the cropped image
                #cropped_image.show()
                cropped_filename = f'{item_key}_{image_index}_{annotation_index}.png'
                save_directory = './cropped'
                save_path = os.path.join(save_directory, cropped_filename)

                cropped_image.save(save_path)
