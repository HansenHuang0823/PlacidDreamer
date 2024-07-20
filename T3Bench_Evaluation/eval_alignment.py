import openai
import backoff
import numpy as np

openai.api_key = "PLEASE USE YOUR OWN API KEY"
MODEL = "gpt-4"

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError))
def predict(prompt, temperature):
    message = openai.ChatCompletion.create(
        model = MODEL,
        temperature = temperature,
        messages = [
                # {"role": "system", "content": prompt}
                {"role": "user", "content": prompt}
            ]
    )
    # print(message)
    return message["choices"][0]["message"]["content"]

with open("./prompt_single.txt", "r") as f:
    prompt_list = f.readlines()

for i in range(len(prompt_list)):
    prompt = prompt_list[i].strip()
    with open(f"./blip_results/{str(i)}.txt", "r") as f:
        texts = f.readlines()
    for j in range(len(texts)):
        texts[j] = texts[j].strip()
    prompt_input = 'Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows:\n\n'
    for idx, txt in enumerate(texts):
        prompt_input += f'view{idx+1}: '
        prompt_input += txt
        prompt_input += '\n'
    prompt_input += '\nAvoid describing background, surface, and posture. The caption should be:'
    res = predict(prompt_input, 0)
    print(res)

    with open(f'./caption_results.txt', 'a+') as f:
        f.write(prompt + ':' + res + '\n')

mean_score = 0
output = ''

prompt_to_gpt4_ori = '''You are an assessment expert responsible for prompt-prediction pairs. Your task is to score the prediction according to the following requirements:

1. Evaluate the recall, or how well the prediction covers the information in the prompt. If the prediction contains information that does not appear in the prompt, it should not be considered as bad.
2. If the prediction contains correct information about color or features in the prompt, you should also consider raising your score.
3. Assign a score between 1 and 5, with 5 being the highest. Do not provide a complete answer; give the score in the format: 3

'''

with open(f'./caption_results.txt', "r") as f:
    lines = f.readlines()

def process_text(text):
    if text[-1] == '.':
        text = text[:-1]
    if text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    if text[0] == '\'' and text[-1] == '\'':
        text = text[1:-1]
    if text[-1] == '.':
        text = text[:-1]
    return text

for line in lines:
    line = line.strip()
    prompt, text = line.split(':')
    text = process_text(text)

    prompt_to_gpt4 = prompt_to_gpt4_ori
    prompt_to_gpt4 += 'Prompt: ' + prompt + '\n'
    prompt_to_gpt4 += 'Prediction: ' + text
    print(prompt_to_gpt4)
    res = predict(prompt_to_gpt4, 0)

    print(res)
    mean_score += np.round(float(res)) / len(lines)
    output += f'{np.round(float(res)):.0f}\t\t{prompt}\n'

print("Alignment Score:", mean_score)

with open(f'alignment_results.txt', 'a+') as f:
    f.write(output)