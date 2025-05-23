import json

import datasets
import tqdm
from websockets.asyncio.async_timeout import final

if __name__ == '__main__':
    project_path=""
    dataset=datasets.load_dataset(f"{project_path}/data/tulu-v2-sft-mixture",split='train')
    final_data=[]
    for d in tqdm.tqdm(dataset):
        text=" ".join([c['content'] for c in d['messages']])
        if len(text.split(" "))>512*1.5:
            continue
        final_data.append({"messages":d['messages']})

    if len(final_data)>300000:
        final_data=final[:300000]
    print(len(final_data))
    with open(f"{project_path}/data/ubmoe_train/train.json",'w') as file:
        json.dump(final_data,file,indent=4,ensure_ascii=False)
