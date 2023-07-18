import argparse
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4B", help="The model id")
    parser.add_argument("--model_path", type=str, default="./pythia", help="The path to save the model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=128)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    prompts = ["Hello, I'm a language model, " for _ in range(args.batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(prompts, return_tensors="pt")
    start = time.time()
    with torch.inference_mode():
        generated_sequence = model.generate(**inputs,
                                            min_length=args.seq_length,
                                            max_length=args.seq_length)
        print([tokenizer.decode(tok) for tok in generated_sequence])
    end = time.time()
    print(f"Outputs generated using AWS sampling loop in {end -start:.4f} s")
