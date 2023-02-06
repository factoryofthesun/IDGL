from transformers import T5Tokenizer, T5Model
from pdb import set_trace
tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir = "./cache")
model = T5Model.from_pretrained("t5-small", cache_dir = "./cache")
input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
outputs = model(input_ids=input_ids)
last_hidden_states = outputs.last_hidden_state
set_trace()
print("set ")
