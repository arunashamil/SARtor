import json
import pandas as pd

def json2csv(input):
    with open(input, 'r') as f:
        data = json.load(f)
    
    rows = []
    for item in data["images"]:
        for sentence in item["sentences"]:
            rows.append([item["filename"], sentence["raw"]])

    return(pd.DataFrame(rows, columns=["Image Name", "Caption"]))
