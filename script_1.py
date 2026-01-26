import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "test", "infer"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--prompt", type=str, default="Hello")

    args = parser.parse_args()

    if args.mode == "train":
        from src.Training.training import Trainer
        from src.data.pipeline import get_dataloader
        from src.models.Model import Model
        from configs.train import TrainConfig
        train_config = TrainConfig()
        model_config = ModelConfig()

        model = Model(model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

        dataloader, _, _ = get_dataloader(
            batch_size=train_config.batch_size,
            context_length=train_config.context_length,
            chunk_tokens=train_config.chunk_tokens,
        )

        trainer = Trainer(model, optimizer, train_config, train_config.ckpt_dir)
        trainer.train(dataloader)

    elif args.mode == "test":
        from testing.test import test
        test(checkpoint=args.checkpoint)

    elif args.mode == "infer":

        import torch
        from src.inference.interactive import Inference
        from src.models.Model import Model
        from src.tokenizer.Tokenizer import Tokenizer
        from configs.model import ModelConfig
        from configs.inference import InferenceConfig

        infer_config = InferenceConfig()
        model_config = ModelConfig()

        model = Model(model_config).to(model_config.device)

        ckpt = torch.load(args.checkpoint, map_location=model_config.device)
        model.load_state_dict(ckpt["model"])
        print(f"checkpoint loaded at {args.checkpoint}")
        model.eval()

        tokenizer = Tokenizer()

        infer = Inference(
            model=model,
            tokenizer=tokenizer,
            max_tokens=infer_config.max_new_token,
            context_size=model_config.context_len,
        )

        with torch.no_grad():
            out = infer.run(args.prompt)
if __name__ == "__main__":
    main()