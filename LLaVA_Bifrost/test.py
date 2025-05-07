from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.6-34b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "liuhaotian/llava-v1.6-34b"
# prompt = """ 
#     Place the bag on the chair and behind
#     the bed, return depth_scale values [min, max] of the center point.
# """
prompt = """
Place a bag naturally on the chair in the attached image.

Then, return one thing **only**:

1. The bounding box of the placed object in [x, y, w, h] format. All four numbers must be between 0 and 1. Make sure that x + w is less than 1 and y + h is less than 1.

The full output must be formatted like this:
[<x>, <y>, <w>, <h>]

Return only one list with exactly 4 numbers. No extra numbers. No words.

"""

# prompt = """
# Place a bag naturally on the chair in the attached image.

# Then, return one thing **only**:

# 1. The **depth value at the center of the image**, as a **range** [min, max], again with two values between 0 and 1.

# The full output must be formatted like this:
# [<min>, <max>]

# Return only one list with exactly 2 numbers. No extra numbers. No words.

# """

image_file = "/home/ec2-user/dev/Bifrost/Main_Bifrost/examples/TEST/Input/background.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0.2,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)