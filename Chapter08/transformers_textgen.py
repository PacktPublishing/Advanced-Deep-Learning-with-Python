import torch
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate pre-trained model-specific tokenizer and the model itself
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103').to(device)

# Initial input sequence
text = "The company was founded in"
tokens_tensor = \
    torch.tensor(tokenizer.encode(text)) \
        .unsqueeze(0) \
        .to(device)

mems = None  # recurrence mechanism

predicted_tokens = list()
for i in range(50):  # stop at 50 predicted tokens
    # Generate predictions
    predictions, mems = model(tokens_tensor, mems=mems)

    # Get most probable word index
    predicted_index = torch.topk(predictions[0, -1, :], 1)[1]

    # Extract the word from the index
    predicted_token = tokenizer.decode(predicted_index)

    # break if [EOS] reached
    if predicted_token == tokenizer.eos_token:
        break

    # Store the current token
    predicted_tokens.append(predicted_token)

    # Append new token to the existing sequence
    tokens_tensor = torch.cat((tokens_tensor, predicted_index.unsqueeze(1)), dim=1)

print('Initial sequence: ' + text)
print('Predicted output: ' + " ".join(predicted_tokens))
