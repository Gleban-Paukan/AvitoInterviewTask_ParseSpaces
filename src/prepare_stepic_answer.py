import pandas as pd

df = pd.read_json('data/pred8.jsonl', lines=True)
ans = df['splits']
ans.to_csv('attempt_4_avito_output.csv', header='predicted_positions')