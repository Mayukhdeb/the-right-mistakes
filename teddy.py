from magma import Magma
from magma.image_input import ImageInput
from utils.resnet50_inference import Resnet50Inference

## load magma into gpu:0 (you'll need a relatively big gpu)
model = Magma.from_checkpoint(
    config_path="configs/MAGMA_v1.yml",
    checkpoint_path="./mp_rank_00_model_states.pt",
    device="cuda:0",
)

inputs = [
    ## supports urls and path/to/image
    ImageInput("images/teddy.jpg"),
    "a picture of",
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
print("Completion from MAGMA:", output[0])  ## a bear costume

## Resnet50 Inference
inference = Resnet50Inference(
    device="cuda:0", imagenet_classes_filename="imagenet_classes.txt"
)

result = inference.run(filename="images/teddy.jpg", num_top_categories=1)

print(
    f"Prediction from Resnet50:\nclass: {result['categories'][0]}\nprobability: {result['probabilities'][0]}"
)