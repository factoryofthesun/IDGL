from pdb import set_trace
from transformers import AutoTokenizer, AutoModel
import torch
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
#sentences = ['This is an example sentence', 'Each sentence is converted']
sentences = [ 'Each sentence is converted']
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', cache_dir = "./local_dir")
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens',cache_dir = "./local_dir")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddingsi
model.train()
model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

set_trace()

x = sentence_embeddings.sum()
x.backward()
list(model.parameters())[-1]
for i in list(model.parameters()):
    print(i)
    set_trace()
print("Sentence embeddings:")
print(sentence_embeddings)

