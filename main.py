# Imports
from my_package.model import InstanceSegmentationModel
from my_package.data import Dataset
from my_package.analysis import plot_visualization, random_colour_masks
from my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

def experiment(annotation_file, segmentator, transforms, outputs):
    '''
        Function to perform the desired experiments

        Arguments:
        annotation_file: Path to annotation file
        segmentator: The object segmentator
        transforms: List of transformation classes
        outputs: path of the output folder to store the images
    '''

    # Create the instance of the dataset.
    dataset = Dataset(annotation_file=annotation_file, transforms=None)

    # Iterate over all data items.
    len_dataset = len(dataset)
    for i in range(len_dataset):
        # Get the predictions from the segmentator.
        print("Predicting the {}th image".format(i))
        image_pred = dataset[i]
        pred_boxes, pred_masks, pred_classes, pred_scores = segmentator(image_pred["image"])
        # Draw the boxes on the image and save them.
        image = image_pred["image"]
        output_path = outputs + "all_images/out" + str(i) + ".jpg"
        plot_visualization(image, pred_boxes, pred_masks, pred_classes, pred_scores, output_path)
 
     
    # Do the required analysis experiments.
    Roll_number_last_digit = 5
    print("Analysing the {}th Image".format(Roll_number_last_digit))
    image_pred = dataset[Roll_number_last_digit]
    image = image_pred["image"]
    output_path_analysis = outputs + "image_analysis/output{}.jpg".format(Roll_number_last_digit)
    pred_boxes, pred_masks, pred_classes, pred_scores = segmentator(image)
    img = plot_visualization(image, pred_boxes, pred_masks, pred_classes, pred_scores, output_path_analysis)
    plt.subplot(4, 4, 1)
    #plt.title("Image")
    plt.imshow(img)


    print("Horizontally flipped original image along with the predicted bounding boxes.")
    flip_data = Dataset(annotation_file=annotation_file,transforms=[transforms[0]()])
    flip_dict = flip_data[Roll_number_last_digit]
    pred_boxes, pred_masks, pred_classes, pred_scores = segmentator(flip_dict["image"])
    image = flip_dict["image"]
    output_path_flip = outputs + "image_analysis/output_flip{}.jpg".format(Roll_number_last_digit)
    flip_img = plot_visualization(image, pred_boxes, pred_masks, pred_classes, pred_scores, output_path_flip)
    plt.subplot(4, 4, 2)
    #plt.title("Image")
    plt.imshow(flip_img)

    print("Blurred image (with some degree of blurring) along with the predicted bounding boxes.")
    blur_data = Dataset(annotation_file, [transforms[1](2)])
    blur_dict = blur_data[Roll_number_last_digit]
    pred_boxes, pred_masks, pred_classes, pred_scores = segmentator(blur_dict["image"])
    image = blur_dict["image"]
    output_path_blur = outputs + "image_analysis/output_blur{}.jpg".format(Roll_number_last_digit)
    blur_img = plot_visualization(image, pred_boxes, pred_masks, pred_classes, pred_scores, output_path_blur)
    plt.subplot(4, 4, 3)
    #plt.title("Image")
    plt.imshow(blur_img)
    #
    print("Twice Rescaled image (2X scaled) along with the predicted bounding boxes")
    rescale1_data = Dataset(annotation_file, [transforms[3]((2 * dataset[Roll_number_last_digit]["image"].shape[2],
                                                             2 * dataset[Roll_number_last_digit]["image"].shape[1]))])
    rescale1_dict = rescale1_data[Roll_number_last_digit]
    pred_boxes, pred_masks, pred_classes, pred_scores = segmentator(rescale1_dict["image"])
    image = rescale1_dict["image"]
    output_path_rescale1 = outputs + "image_analysis/output_rescale2X{}.jpg".format(Roll_number_last_digit)
    rescale1_img = plot_visualization(image, pred_boxes, pred_masks, pred_classes, pred_scores, output_path_rescale1)
    plt.subplot(4, 4, 4)
    #plt.title("Image")
    plt.imshow(rescale1_img)

    print("Half Rescaled image (0.5X scaled) along with the predicted bounding boxes")
    rescale2_data = Dataset(annotation_file, [transforms[3]((int(dataset[Roll_number_last_digit]["image"].shape[2] / 2),
                                                             int(dataset[Roll_number_last_digit]["image"].shape[
                                                                     1] / 2)))])
    rescale2_dict = rescale2_data[Roll_number_last_digit]
    pred_boxes, pred_masks, pred_classes, pred_scores = segmentator(rescale2_dict["image"])
    image = rescale2_dict["image"]
    output_path_rescale2 = outputs + "image_analysis/output_rescale.5X{}.jpg".format(Roll_number_last_digit)
    rescale2_img = plot_visualization(image, pred_boxes, pred_masks, pred_classes, pred_scores, output_path_rescale2)
    plt.subplot(4, 4, 5)
    #plt.title("Image")
    plt.imshow(rescale2_img)
    #
    print("90 degree right rotated image along with the predicted bounding boxes")
    rotate1_data = Dataset(annotation_file, [transforms[2](-90)])
    rotate1_dict = rotate1_data[Roll_number_last_digit]
    pred_boxes, pred_masks, pred_classes, pred_scores = segmentator(rotate1_dict["image"])
    image =rotate1_dict["image"]
    output_path_rotate = outputs + "image_analysis/output_rotate90{}.jpg".format(Roll_number_last_digit)
    rotate1_img = plot_visualization(image, pred_boxes, pred_masks, pred_classes, pred_scores, output_path_rotate)
    plt.subplot(4, 4, 6)
    #plt.title("Image")
    plt.imshow(rotate1_img)

    print("45 degree left rotated image along with the predicted bounding boxes")
    rotate1_data = Dataset(annotation_file, [transforms[2](45)])
    rotate1_dict = rotate1_data[Roll_number_last_digit]
    pred_boxes, pred_masks, pred_classes, pred_scores = segmentator(rotate1_dict["image"])
    image = rotate1_dict["image"]
    output_path_rotate = outputs + "image_analysis/output_rotate45{}.jpg".format(Roll_number_last_digit)
    rotate1_img = plot_visualization(image, pred_boxes, pred_masks, pred_classes, pred_scores, output_path_rotate)
    plt.subplot(4, 4, 7)
    #plt.title("Image")
    plt.imshow(rotate1_img)


    save_path_file = outputs + 'image_analysis/analysis.png'
    plt.savefig(save_path_file)
    plt.show()

def instance_segmentation_api(img_path, rect_th=3, text_size=3, text_th=3):

      
      img= Image.open(img_path)
      
      np_img = np.array(img)
      a=InstanceSegmentationModel()
      pred_masks, pred_boxes, pred_classes = a(np_img)
      img = cv2.imread(img_path)

      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      for i in range(len(pred_masks)):

        rgb_mask = random_colour_masks(pred_masks[i])

        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

        cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=(0, 255, 0), thickness=rect_th)
         
        cv2.putText(img,pred_classes[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

      plt.figure(figsize=(20,30))

      plt.imshow(img)
      plt.xticks([])
      plt.yticks([])
      plt.show()


def main():
    segmentator = InstanceSegmentationModel()
    experiment('/Users/mirmohammadwasif/Python_DS_Assignment/data/annotations.jsonl', segmentator, [FlipImage, BlurImage, RotateImage, RescaleImage, CropImage],
               '/Users/mirmohammadwasif/Python_DS_Assignment/outputs/')  # Sample arguments to call experiment()
    instance_segmentation_api('/Users/mirmohammadwasif/Python_DS_Assignment/data/imgs/5.jpeg')

if __name__ == '__main__':
    main()

# learn how to structure the code
# where should the setup.py be and __init__.py be?