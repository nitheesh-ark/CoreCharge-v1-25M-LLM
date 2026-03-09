from transformers import AutoTokenizer


class Tokenizer:
    def __init__(
        self,
        model_name: str = "pszemraj/bytebpe-tokenizer-32k-mlm-uncased"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        self.tokenizer.add_special_tokens({
            "unk_token": "<UNK>",
            "pad_token": "<PAD>",
            "additional_special_tokens": ["<USER>", "<BOT>"],
        })

       


    def encode(self, text: str) -> list[int]:

        return self.tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]

    def decode(self, token_ids: list[int]) -> str:

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True
        )
    
    def __len__(self) -> int:
        return self.tokenizer.vocab_size    


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(len(tokenizer))

    text = "this is raw text"
    ids = tokenizer.encode(text)

    print("Token IDs:", ids)
    print("Decoded :", tokenizer.decode(ids))
