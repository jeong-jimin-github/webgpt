import torch
from transformers import T5Tokenizer, AutoModelForCausalLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer.do_lower_case = True
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
model.to(device)
model.eval()

def generate_reply(inp, num_gen=5):
    input_ids = tokenizer.encode(inp, return_tensors='pt').to(device)
    out = model.generate(input_ids, do_sample=True, max_new_tokens=128, num_return_sequences=num_gen, temperature=1,
                         top_p=0.95, top_k=50, bad_words_ids=[[1], [5]], no_repeat_ngram_size=2)

    return(tokenizer.batch_decode(out)[0].replace("</s>", "").replace("</d>", ""))

if __name__ == '__main__':
    text = "パソコンの中央処理装置は、"
    generate_reply(text)
