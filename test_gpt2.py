import argparse
import os
import time
import torch
from tempfile import TemporaryDirectory
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'


def export_neuron_model(model_id, batch_size, seq_length, tp_degree, amp, save_dir):

    # Load and save the CPU model
    model_cpu = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    chkpt_dir = save_dir + "/checkpoint"
    save_pretrained_split(model_cpu, chkpt_dir)


    # Create and compile the Neuron model
    model_neuron = GPT2ForSampling.from_pretrained(chkpt_dir,
                                                   batch_size=batch_size,
                                                   tp_degree=tp_degree,
                                                   n_positions=seq_length,
                                                   amp=amp,
                                                   unroll=None)
    model_neuron.to_neuron()

    compiled_dir = save_dir + "/compiled"
    model_neuron._save_compiled_artifacts(compiled_dir)
    model_cpu.config.save_pretrained(save_dir)
    config = AutoConfig.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)
    return model_neuron, config, tokenizer


def load_neuron_model(save_dir, batch_size, seq_length, tp_degree, amp):
    # Load the Neuron model
    chkpt_dir = save_dir + "/checkpoint"
    model_neuron = GPT2ForSampling.from_pretrained(chkpt_dir,
                                                   batch_size=batch_size,
                                                   tp_degree=tp_degree,
                                                   n_positions=seq_length,
                                                   amp=amp,
                                                   unroll=None)
    # Load the compiled Neuron artifacts
    compiled_dir = save_dir + "/compiled"
    model_neuron._load_compiled_artifacts(compiled_dir)
    model_neuron.to_neuron() # Load the model weights but skip compilation
    config = AutoConfig.from_pretrained(chkpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    return model_neuron, config, tokenizer


def test_sample(model_neuron, tokenizer, prompts, seq_length):
    # Encode text and generate using AWS sampling loop
    encoded_text = tokenizer(prompts, return_tensors="pt")
    start = time.time()
    with torch.inference_mode():
        generated_sequence = model_neuron.sample(encoded_text.input_ids, sequence_length=seq_length)
        print([tokenizer.decode(tok) for tok in generated_sequence])
    end = time.time()
    print(f"Outputs generated using AWS sampling loop in {end -start:.4f} s")


def test_generate(config, model_neuron, tokenizer, prompts, seq_length):
    # Use the `HuggingFaceGenerationModelAdapter` to access the generate API
    model = HuggingFaceGenerationModelAdapter(config, model_neuron)
    # Encode text and generate using AWS sampling loop
    tokens = tokenizer(prompts, return_tensors="pt")
    start = time.time()
    with torch.inference_mode():
        # Run inference using temperature
        model.reset_generation()
        sample_output = model.generate(
            **tokens,
            do_sample=True,
            max_length=seq_length,
            top_k=50,
        )
        print([tokenizer.decode(tok) for tok in sample_output])
    end = time.time()
    print(f"Outputs generated using HF generate in {end - start:.4f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The HF Hub model id or a local directory")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=1000)
    parser.add_argument("--tp_degree", type=int, default=2)
    parser.add_argument("--amp", type=str, default="f32", help="One of f32, f16, bf16")
    parser.add_argument("--save_dir", type=str, help="The save directory")
    parser.add_argument("--compare", action="store_true", help="Compare with the genuine transformers model on CPU.")
    args = parser.parse_args()
    if not os.path.isdir(args.model):
        if args.save_dir is None:
            raise ValueError("You need to specify a save_dir when exporting a model from the hub")
        # Export llm model
        model, config, tokenizer = export_neuron_model(args.model,
                                                       args.batch_size,
                                                       args.seq_length,
                                                       args.tp_degree,
                                                       args.amp,
                                                       args.save_dir)
    else:
        model, config, tokenizer = load_neuron_model(args.model,
                                                     args.batch_size,
                                                     args.seq_length,
                                                     args.tp_degree,
                                                     args.amp)
    # Get an example input
    prompt_text = "Hello, I'm a language model,"
    # We need to replicate the text if batch_size is not 1
    prompts = [prompt_text for _ in range(args.batch_size)]
    # Test model using AWS sample loop
    test_sample(model, tokenizer, prompts, args.seq_length)
    # Test model using HF generate
    test_generate(config, model, tokenizer, prompts, args.seq_length)
