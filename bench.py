import os
import sglang as sgl

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    runtime = sgl.Runtime(model_path=path, tokenizer_path=path)
    sgl.set_default_backend(runtime)

    @sgl.function
    def text_completion(s, question):
        s += question + sgl.gen("answer")

    prompts = [
        {"question": "Hello, my name is"},
        {"question": "The president of the United States is"},
        {"question": "The capital of France is"},
        {"question": "The future of AI is"},
    ]
    states = text_completion.run_batch(prompts, progress_bar=True)

    for prompt, state in zip(prompts, states):
        print(f"Prompt: {prompt['question']!r}")
        print(f"Generated: {state['answer']!r}\n")


if __name__ == "__main__":
    main()