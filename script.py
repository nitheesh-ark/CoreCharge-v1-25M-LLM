import argparse
import torch

from configs.train import TrainConfig
from configs.model import ModelConfig
from configs.inference import InferenceConfig

# from src.Training.training import Trainer
from src.models.Model import Model
from src.tokenizer.Tokenizer import Tokenizer
from src.inference.interactive import Inference

# from testing.test import test
# from src.Training.train import main as train_main  # 🔹 import the training entrypoint


def main():
    parser = argparse.ArgumentParser(description="Model Control Script")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test", "infer"],
        help="Mode to run: train | test | infer"
    )

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--chunk_id", type=int, default=None)
    args = parser.parse_args()


    if args.mode == "train":

        def train():
            print(f"\n Training pipeline is not open source due to an unstable codebase.\n")
        train()



    elif args.mode == "test":
        
        def test():
            print("\nThe codebase is not stable yet, so it is currently closed-source. It will be opened in upcoming releases.\n")
        test()


    elif args.mode == "infer":
        model_config = ModelConfig()
        infer_config = InferenceConfig()

        model = Model(model_config).to(model_config.device)
        model.eval()

        if args.checkpoint is not None:
            state = torch.load(args.checkpoint, map_location=model_config.device)
            model.load_state_dict(state["model"])

        tokenizer = Tokenizer()

        infer = Inference(
            model=model,
            tokenizer=tokenizer,
            config=infer_config  # 🔹 pass instance, not class
        )

        output = infer.run(args.prompt)
        print("system :::", output)


if __name__ == "__main__":
    main()
