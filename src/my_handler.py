# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler
import torch
import os

from model import Net
from config import ConfigL
import requests
from PIL import Image
import io

class MyHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        config = ConfigL()

        self.model = Net(
            clip_model=config.clip_model,
            text_model=config.text_model,
            ep_len=config.ep_len,
            num_layers=config.num_layers, 
            n_heads=config.n_heads, 
            forward_expansion=config.forward_expansion, 
            dropout=config.dropout, 
            max_len=config.max_len,
            device=self.device
        )
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        image = Image.open(io.BytesIO(data[0].get("body")))  
        
        return image

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        caption, _ = self.model(data, 1.0)
        print(caption)
        
        return caption

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.

        Returns:
            List: The post process function returns a list of the predicted output.
        """
        data_list = []
        data_list.append(data)

        return data_list

    # def handle(self, data, context):
    #     """
    #     Invoke by TorchServe for prediction request.
    #     Do pre-processing of data, prediction using model and postprocessing of prediciton output
    #     :param data: Input data for prediction
    #     :param context: Initial context contains model server system properties.
    #     :return: prediction output
    #     """
    #     image = Image.open(io.BytesIO(data[0].get("body")))        
    #     caption, _ = self.model(image, 1.0)
        
    #     return caption