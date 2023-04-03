import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, state_dict=torch.load('.\model.pth', map_location=torch.device('cpu')))

# Take input from user
article = input("Enter the article to summarize: ")

# Preprocess input
input_ids = tokenizer.encode(article, return_tensors='pt')


# Generate summary
summary_ids = model.generate(input_ids=input_ids, max_length=2048, num_beams=3, repetition_penalty=1.0, length_penalty=3, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print summary
print(summary)







