import torch
from configs.train import TrainConfig
from configs.model import ModelConfig
from src.models.Model import Model
from src.data.dataloader import build_dataloader
from src.Training.training import Trainer

def main():

    config = TrainConfig()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_config = ModelConfig()
    model = Model(model_config).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


    dataloader = build_dataloader(
        token_folder=config.token_folder,
        chunk_id=config.chunk_id,       # uses chunk_id from TrainConfig
        seq_len=config.context_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        ckpt_dir=config.ckpt_dir,
    )


    trainer.train(dataloader)


if __name__ == "__main__":
    main()
