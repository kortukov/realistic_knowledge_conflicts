import justext
import os
import pandas as pd
import requests

RAW_FRESHQA_URL = "https://docs.google.com/spreadsheet/ccc?key=1V6nIxVTI9tqZ-wfgK-uFuUPiGEa1Zmnz53OeGbaNtO0&output=csv"
RAW_FRESHQA_PATH = "data/freshqa/raw_freshqa.csv"
CHANGING_FRESHQA_PATH = "data/freshqa/changing_freshqa.parquet"
ICL_FRESHQA_PATH = "data/freshqa/icl_freshqa.parquet"


def save_csv():
    print("Downloading FreshQA CSV")
    response = requests.get(RAW_FRESHQA_URL)
    assert response.status_code == 200, f"Failed to download FreshQA CSV, response {response.status_code}"

    print("Saving FreshQA CSV")
    os.makedirs(os.path.dirname(RAW_FRESHQA_PATH), exist_ok=True)
    with open(RAW_FRESHQA_PATH, "wb") as f:
        f.write(response.content)


def get_text(urls, idx=[0]):

    # Is this neat or is this horrible? I like it. 
    print(idx[0])
    idx[0]+=1 

    total_text = ""

    if not urls or pd.isna(urls):
        return total_text
    urls = urls.split('\n')

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
            url_text = "\n".join([p.text for p in paragraphs if not p.is_boilerplate])
            total_text += url_text + "\n"
        except:
            print(f"Error with url: {url}")
    
    return total_text


def process_freshqa():
    print("Processing FreshQA data")

    print("Filling in answers")
    freshqa = pd.read_csv(RAW_FRESHQA_PATH, skiprows=[0,1])
    answer_columns = [col for col in freshqa.columns if 'answer' in col]
    freshqa['answers'] = freshqa[answer_columns].apply(lambda row: [a for a in row if pd.notna(a)], axis=1)

    print("Filling in context from the web source") 
    freshqa['context'] = freshqa.source.apply(get_text)

    no_empty_context = freshqa[freshqa.context.apply(lambda x: len(x) > 0)]

    icl_examples = no_empty_context.sample(10, random_state=42)

    full_freshqa = no_empty_context[~no_empty_context.index.isin(icl_examples.index)]
    freshqa_slow = full_freshqa[full_freshqa.fact_type == "slow-changing"]
    freshqa_fast = full_freshqa[full_freshqa.fact_type == "fast-changing"]

    freshqa_changing = pd.concat([freshqa_slow, freshqa_fast])

    print("Saving data")
    icl_examples.to_parquet(ICL_FRESHQA_PATH)
    freshqa_changing.to_parquet(CHANGING_FRESHQA_PATH)

if __name__ == "__main__":
    save_csv()

    process_freshqa()