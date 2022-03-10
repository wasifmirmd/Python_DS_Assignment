# # Imports
import numpy as np
from PIL import Image
import json


class Dataset(object):
#     '''
#         A class for the dataset that will return data items as per the given index
#     '''

    def __init__(self, annotation_file, transforms=None):
#         '''
#             Arguments:
#             annotation_file: path to the annotation file
#             transforms: list of transforms (class instances)
#                         For instance, [<class 'RandomCrop'>, <class 'Rotate'>]
#         '''
        self.annotation_file = annotation_file
        self.transforms = transforms
        with open(self.annotation_file, 'r') as f:
            json_list = list(f)
        json_data_list = []
        for i in json_list:
            json_data_list.append(json.loads(i))
        self.json_list = json_data_list

    def __len__(self):
#         '''
#             return the number of data points in the dataset
#         '''
        with open(self.annotation_file, 'r') as f:
            json_list = list(f)
        json_data_list = []
        for i in json_list:
            json_data_list.append(json.loads(i))
        return len(self.json_list)

    def __getitem__(self, idx):
#         '''
#             return the dataset element for the index: "idx"
#             Arguments:
#                 idx: index of the data element.

#             Returns: A dictionary with:
#                 image: image (in the form of a numpy array) (shape: (3, H, W))
#                 gt_bboxes: N X 5 array where N is the number of bounding boxes, each
#                             consisting of [class, x1, y1, x2, y2]
#                             x1 and x2 lie between 0 and width of the image,
#                             y1 and y2 lie between 0 and height of the image.

#             You need to do the following,
#             1. Extract the correct annotation using the idx provided.
#             2. Read the image and convert it into a numpy array (wont be necessary
#                 with some libraries). The shape of the array would be (3, H, W).
#             3. Scale the values in the array to be with [0, 1].
#             4. Create a dictonary with both the image and annotations
#             4. Perform the desired transformations.
#             5. Return the transformed image and annotations as specified.
#         '''
        with open(self.annotation_file, 'r') as f:
            json_list = list(f)
        json_data_list = []
        for i in json_list:
            json_data_list.append(json.loads(i))
        data_at_idx = json_data_list[idx]
        image_path = '/Users/mirmohammadwasif/Python_DS_Assignment/data/' + data_at_idx['img_fn']
        image_idx = Image.open(image_path)
        #image_idx=Image.open(self.annotation_file[:self.annotation_file.find('annotations.jsonl')] + data_at_idx["img_fn"])
        image_np_array = np.asarray(image_idx)
        if self.transforms:
          for transformation in self.transforms:
              image_np_array = transformation(image_np_array)

        image_np_array = image_np_array / 255
        image_np_array = image_np_array.transpose((2, 0, 1))
        res_dict = dict()
        res_dict["image"] = image_np_array
        gtbboxes = []
        bboxes = data_at_idx["bboxes"]
        for boxes in bboxes:
            gtbboxes.append([boxes["category"], boxes['bbox'][0], boxes['bbox'][1], boxes['bbox'][2], boxes['bbox'][3]])

        res_dict["gtbboxes"] = gtbboxes
        return res_dict
# # Imports
# import json
# from PIL import Image
# import numpy as np


# class Dataset(object):
#     """
#         A class for the dataset that will return data items as per the given index
#     """

#     def __init__(self, annotation_file, transforms=None):
#         """
#             Arguments:
#             annotation_file: path to the annotation file
#             transforms: list of transforms (class instances)
#                         For instance, [<class 'RandomCrop'>, <class 'Rotate'>]
#         """
#         # Code Here
#         self.annotationFile = annotation_file
#         self.allTransforms = transforms
#         with open(self.annotationFile, 'r') as annotationFile:
#             datasetJson = list(annotationFile)
#         self.datasetJson = []
#         for jsonStr in datasetJson:
#             self.datasetJson.append(json.loads(jsonStr))

#     def __len__(self):
#         """
#             return the number of data points in the dataset
#         """
#         return len(self.datasetJson)

#     def __getitem__(self, idx):
#         """
#             return the dataset element for the index: "idx"
#             Arguments:
#                 idx: index of the data element.

#             Returns: A dictionary with:
#                 image: image (in the form of a numpy array) (shape: (3, H, W))
#                 gt_bboxes: N X 5 array where N is the number of bounding boxes, each
#                             consisting of [class, x1, y1, x2, y2]
#                             x1 and x2 lie between 0 and width of the image,
#                             y1 and y2 lie between 0 and height of the image.

#             You need to do the following,
#             1. Extract the correct annotation using the idx provided.
#             2. Read the image and convert it into a numpy array (wont be necessary
#                 with some libraries). The shape of the array would be (3, H, W).
#             3. Scale the values in the array to be with [0, 1].
#             4. Create a dictionary with both the image and annotations
#             4. Perform the desired transformations.
#             5. Return the transformed image and annotations as specified.
#         """
#         result = dict()
#         dataSelected = self.datasetJson[idx]
#         image = Image.open(self.annotationFile[:self.annotationFile.find('annotations.jsonl')] + dataSelected["img_fn"])
#         imageArray = np.asarray(image)

#         if self.allTransforms:
#             for transformation in self.allTransforms:
#                 imageArray = transformation(imageArray)

#         imageArray = imageArray/255
#         imageArray = imageArray.transpose((2, 0, 1))
#         result["image"] = imageArray
#         gtbboxes = []
#         bboxes = dataSelected["bboxes"]
#         for i in bboxes:
#             gtbboxes.append([i["category"], i['bbox'][0], i['bbox'][1], i['bbox'][2], i['bbox'][3]])
#         result["gt_bboxes"] = gtbboxes
#         return result