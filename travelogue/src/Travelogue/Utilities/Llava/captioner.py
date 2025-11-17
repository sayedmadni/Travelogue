#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
from svlearn_vlu import config
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import os
import json
from tqdm import tqdm

#  -------------------------------------------------------------------------------------------------
class LlavaCaptioner:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        """
        Initializes the LlavaCaptioner with a pre-trained LLaVA model and its processor.

        Args:
            model_name (str): The name of the LLaVA model to load. Default is
            "llava-hf/llava-1.5-7b-hf".
        """
        # Set device priority: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Configure quantization for efficient inference (only for CUDA)
        if torch.cuda.is_available():
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            self.quant_config = None

        # Load the processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.float16,  # Use half-precision for memory efficiency
            "device_map": "auto",  # Automatically map the model to available hardware
        }
        
        # Add quantization config only if available (CUDA only)
        if self.quant_config is not None:
            model_kwargs["quantization_config"] = self.quant_config
            
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

    # -------------------------------------------------------------------------------------------------

    def _generate_caption(self, image, prompt: str = "Describe the image in detail."):
        """
        Generates a caption for the given image using the LLaVA model.

        Args:
            image (PIL.Image): The image to generate a caption for.
            prompt (str): The text prompt to guide the caption generation. Default is
            "Describe the image in detail."

        Returns:
            str: The generated caption (only the assistant's response).
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Prepare the inputs for the model
        inputs = self.processor(images=[image], text=[text_prompt], padding=True, return_tensors="pt").to(
            self.device, torch.float16
        )

        # Generate the response
        generate_ids = self.model.generate(**inputs, max_new_tokens=200)

        # Decode the generated token IDs into text
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

        # Extract only the assistant's response
        if "ASSISTANT:" in output:
            assistant_response = output.split("ASSISTANT:")[1].strip()
            return assistant_response
        else:
            return output  # Fallback if the assistant's response is not found

    # -------------------------------------------------------------------------------------------------

    def generate_captions(self, image_dir, save_json=False, output_file="captions.json"):
        """
        Generates captions for all images in the specified directory and optionally saves them as a JSON file.

        Args:
            image_dir (str): Path to the directory containing images.
            save_json (bool): If True, saves the captions as a JSON file. Default is False.
            output_file (str): Name of the output JSON file. Default is "captions.json".

        Returns:
            dict: A dictionary where keys are image file names and values are their generated captions.
        """
        # Get a list of image files in the directory
        image_files = [
            f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]

        # Load existing captions if the file exists
        if save_json and os.path.exists(output_file):
            with open(output_file, "r") as f:
                all_captions = json.load(f)
        else:
            all_captions = {}

        # Process images with a progress bar
        for image_file in tqdm(image_files, desc="Processing images", unit="image"):
            image_path = os.path.join(image_dir, image_file)

            try:
                # Open the image
                image = Image.open(image_path).convert("RGB")
                caption = self._generate_caption(image)

                # Update the captions dictionary
                image_caption = all_captions.get(image_file, {})
                image_caption["llava_caption"] = caption
                all_captions[image_file] = image_caption

            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                exit()

        # Save captions to a JSON file if requested
        if save_json:
            with open(output_file, "w") as f:
                json.dump(all_captions, f, indent=4)
            print(f"Captions saved to {output_file}")

        return all_captions


#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = config["datasets"]["unsplash"]
    captioner = LlavaCaptioner()
    captions = captioner.generate_captions(data_dir, save_json=True, output_file=f"{data_dir}/captions.json")

    for filename, caption in captions.items():
        print(f"Image: {filename}, Caption: {caption['llava_caption']}")

#  -------------------------------------------------------------------------------------------------
