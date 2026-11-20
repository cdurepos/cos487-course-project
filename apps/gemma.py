"""
The function `load_model` loads the Gemma 4 model and processor, including the assistant model. 
The function `delete_model` deletes the model from memory to free up resources.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import gc
from transformers.cache_utils import DynamicCache


def load_model(model_id: str = "google/gemma-4-E2b-it", assistant_id: str = "google/gemma-4-E2B-it-assistant", use_assistant: bool = True, gpu: int = None) -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoProcessor]:
    """Loads the Gemma 4 model and processor, including the assistant model.

    Args:
        model_id (str, optional): The model identifier. Defaults to "google/gemma-4-E2B-it".
        assistant_id (str, optional): The assistant model identifier. Defaults to "google/gemma-4-E2B-it-assistant".
        use_assistant (bool, optional): Whether to load the assistant model. Defaults to True.
        gpu (int, optional): The GPU index to use; if using assistant model, only one gpu is available. Defaults to auto.

    Returns:
        tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoProcessor]: The loaded target model, assistant model, and processor.
    """

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu}" if gpu is not None else "auto",
    )
    if not use_assistant:
        print("Model loaded successfully.")
        return model, None, processor
    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_id,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu}" if gpu is not None else "auto",
    )
    print("Model and assistant model loaded successfully.")
    return model, assistant_model, processor


def delete_model(model: AutoModelForCausalLM, assistant: AutoModelForCausalLM = None) -> None:
    """Deletes the model from memory.
    
    Args:
        model (AutoModelForCausalLM): The model to be deleted.
        assistant (AutoModelForCausalLM, optional): The assistant model to be deleted. Defaults to None.
    """

    del model
    print("Model deleted from memory.")
    if assistant is not None:
        del assistant
        print("Assistant model deleted from memory.")
    gc.collect()
    torch.cuda.empty_cache()


def generate(messages: list,
            model: AutoModelForCausalLM,
            processor: AutoProcessor,
            assistant: AutoModelForCausalLM = None, 
            temperature: float = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
            max_new_tokens: int = 512,
            batch_size: int = 1) -> str | list[str]:
    """Generates a response from the model based on the provided messages.

    One conversation uses the assistant model. A list of conversations is padded
    and generated ``batch_size`` at a time, without the assistant model, which
    only supports one conversation. The batched return value keeps that order.

    Args:
        messages (list[dict] | list[list[dict]]): One conversation, or a list of them.
        model (AutoModelForCausalLM): The main model to generate the response.
        processor (AutoProcessor): The processor to handle tokenization and decoding.
        assistant (AutoModelForCausalLM, optional): The assistant model to assist in generation. Defaults to None.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        top_p (float, optional): Top-p sampling parameter. Defaults to 0.95.
        top_k (int, optional): Top-k sampling parameter. Defaults to 64.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 512.
        batch_size (int, optional): Conversations per forward pass. Defaults to 1.

    Returns:
        str | list[str]: The generated response, or one per conversation.
    """
    if messages and isinstance(messages[0], list):
        return _generate_batch(
            messages,
            model=model,
            processor=processor,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    custom_cache = DynamicCache()

    if not temperature or temperature < 0:
        temperature = 1.0
    if not top_p or top_p <= 0 or top_p > 1:
        top_p = 0.95
    if not max_new_tokens or max_new_tokens <= 0:
        max_new_tokens = 512

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            past_key_values=custom_cache,
            assistant_model=assistant,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=True
        )

    generated_tokens = outputs[0][input_len:]
    response = processor.decode(generated_tokens, skip_special_tokens=False)
    del outputs, generated_tokens, inputs
    torch.cuda.empty_cache()

    try:
        parsed_response = processor.parse_response(response)

        if isinstance(parsed_response, dict) and "content" in parsed_response:
            return parsed_response["content"]
        elif isinstance(parsed_response, list) and len(parsed_response) > 0:
            last_turn = parsed_response[-1]
            if isinstance(last_turn, dict) and "content" in last_turn:
                return last_turn["content"]

        if parsed_response:
            response = parsed_response
    except Exception:
        pass
    response = response.strip()
    if response.endswith("<turn|>"):
        response = response[:-7]
    return response


def _generate_batch(conversations: list,
                    model: AutoModelForCausalLM,
                    processor: AutoProcessor,
                    temperature: float,
                    top_p: float,
                    top_k: int,
                    max_new_tokens: int,
                    batch_size: int) -> list[str]:
    """Generate one reply per conversation. Does not use the assistant model."""
    if not temperature or temperature < 0:
        temperature = 1.0
    if not top_p or top_p <= 0 or top_p > 1:
        top_p = 0.95
    if not max_new_tokens or max_new_tokens <= 0:
        max_new_tokens = 512
    if batch_size < 1:
        batch_size = 1

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    replies = []
    for start in range(0, len(conversations), batch_size):
        chunk = conversations[start:start + batch_size]
        texts = [
            processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in chunk
        ]
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "use_cache": True,
        }
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        for row in outputs:
            generated = row[input_len:]
            if pad_token_id is not None:
                end = generated.shape[-1]
                while end > 0 and generated[end - 1].item() == pad_token_id:
                    end -= 1
                generated = generated[:end]
            response = processor.decode(generated, skip_special_tokens=False)
            replies.append(_reply_text(processor, response))
        del outputs, inputs
    return replies


def _reply_text(processor: AutoProcessor, response: str) -> str:
    try:
        parsed_response = processor.parse_response(response)

        if isinstance(parsed_response, dict) and "content" in parsed_response:
            return parsed_response["content"]
        elif isinstance(parsed_response, list) and len(parsed_response) > 0:
            last_turn = parsed_response[-1]
            if isinstance(last_turn, dict) and "content" in last_turn:
                return last_turn["content"]

        return parsed_response if parsed_response else response
    except Exception:
        return response


if __name__ == "__main__":
    # Example usage to test the model loading and generation functions
    model, assistant, processor = load_model()
    question = "What is the integral of x^2?"

    answer = generate(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        model=model,
        processor=processor,
        assistant=assistant
    )
    print("Sample Question:", question)
    print("Generated Answer:", answer)