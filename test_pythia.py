import argparse
import time
import torch

from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers_neuronx.gptneox.model import GPTNeoXForSampling
from transformers_neuronx.module import save_pretrained_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4B", help="The model id")
    parser.add_argument("--model_path", type=str, default="./pythia", help="The path to save the model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--tp_degree", type=int, default=2)
    parser.add_argument("--amp", type=str, default="f32", help="One of f32, f16, bf16")
    args = parser.parse_args()

    model = GPTNeoXForCausalLM.from_pretrained(args.model)
    save_pretrained_split(model, args.model_path)

    neuron_model = GPTNeoXForSampling.from_pretrained(args.model_path,
                                                      batch_size=args.batch_size,
                                                      n_positions=args.seq_length,
                                                      tp_degree=args.tp_degree,
                                                      amp=args.amp)
    neuron_model.to_neuron()

    prompts = ["Hello, I'm a language model, " for _ in range(args.batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(prompts, return_tensors="pt")
    start = time.time()
    with torch.inference_mode():
        generated_sequence = neuron_model.sample(inputs.input_ids, sequence_length=args.seq_length)
        print([tokenizer.decode(tok) for tok in generated_sequence])
    end = time.time()
    print(f"Outputs generated using AWS sampling loop in {end -start:.4f} s")
