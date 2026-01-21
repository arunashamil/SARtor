import json
import pandas as pd

def json2csv(input):
    with open(input, 'r') as f:
        data = json.load(f)
    
    rows = []
    for item in data["images"]:
        rows.append(item["sentences"][0]["raw"])

    return(pd.DataFrame(rows))

    