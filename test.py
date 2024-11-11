from mindnlp.transformers import BloomForCausalLM, AutoTokenizer
import mindspore

path = "bigscience/bloomz-1b7"
tokenizer = AutoTokenizer.from_pretrained(path)
model = BloomForCausalLM.from_pretrained(path, ms_dtype=mindspore.float16)


content = "hello!"
inputs = tokenizer(content, return_tensors="ms")
outputs = model.generate(
    inputs.input_ids,
    max_length=50,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)