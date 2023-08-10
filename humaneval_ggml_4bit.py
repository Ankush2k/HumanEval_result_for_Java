from  gpt4all import GPT4ALL
import pandas as pd

model = GPT4ALL(model_name = 'ggml-model_q4_1.bin', model_path = ('.'), allow_download=False)

def get_response(prompt):
    response = model.generate(prompt=prompt, max_tokens=200)
    return response

df = pd.read_json('humaneval_java.jsonl', lines=True)

li = df['prompt'].tolist()
res = []
c=1
for i in li:
    print('Starting ',c)
    res.append(get_response(i))
    c+=1

res_list = [li[i]+res[i] for i in range(len(li))]

df['gen_result'] = res
df['gen_comp_res'] = res_list

df.to_csv('starcoderbase3b-ggml-4bit.csv')