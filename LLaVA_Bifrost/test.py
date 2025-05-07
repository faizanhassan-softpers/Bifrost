from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "liuhaotian/llava-v1.5-7b"
# prompt = """ 
#     Place the bag on the chair of the image. Return: 
#     The bounding box of the object in [x, y, w, h] in the range of [0, 1] format. 
#     The depth value at the center point of the image.
# """
prompt = """
Place a medium-sized bag naturally on the chair in the image.
Return only the following:

The bounding box of the object in normalized [x, y, w, h] format (values between 0 and 1).

Ensure x + w ≤ 1 and y + h ≤ 1.

The depth value at the center of the image.
Format the response exactly as:
[x, y, w, h], [x, y]
"""
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