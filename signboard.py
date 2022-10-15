from magma import Magma
from magma.image_input import ImageInput

## load magma into gpu:0 (you'll need a relatively big gpu)
model = Magma.from_checkpoint(
    config_path="configs/MAGMA_v1.yml",
    checkpoint_path="./mp_rank_00_model_states.pt",
    device="cuda:0",
)

inputs = [
    ## supports urls and path/to/image
    ImageInput("images/sign_original.jpg"),
    "This is a sign that says:",
]

## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(inputs)

## returns a list of length embeddings.shape[0] (batch size)
output = model.generate(
    embeddings=embeddings,
    max_steps=20,
    temperature=0.0001,
    top_k=0,
)
print("Completion from MAGMA on original image:", output[0])


inputs = [
    ## supports urls and path/to/image
    ImageInput("images/sign_modified.jpg"),
    "This is a sign that says:",
]

## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(inputs)

## returns a list of length embeddings.shape[0] (batch size)
output = model.generate(
    embeddings=embeddings,
    max_steps=20,
    temperature=0.0001,
    top_k=0,
)
print("Completion from MAGMA on modified image:", output[0])