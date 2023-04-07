import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

def generate_reply(inp, num_gen=5):
    input_ids = tokenizer.encode(inp, return_tensors='pt').to(device)
    out = model.generate(input_ids, do_sample=True, max_new_tokens=128, num_return_sequences=num_gen, temperature=1,
                         top_p=0.95, top_k=50, bad_words_ids=[[1], [5]], no_repeat_ngram_size=2)

    return(tokenizer.batch_decode(out)[0].replace("</s>", "").replace("</d>", ""))

if __name__ == '__main__':
    text = "The computer's central processing unit is"
    generate_reply(text)
