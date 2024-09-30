import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from preprocess2 import DrugCombinationDataset, load_data
from model_MLP import SNN, evaluate_model
import networkx as nx

# 경로 설정
network_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/protein-protein_network.csv'
drug_protein_path = 'E:/CPI 빅데이터/시너지_서현수집/merged_OB_com_protein.csv'
drug_combinations_path = 'E:/CPI 빅데이터/시너지_서현수집/OB_combination.csv'
drug_fingerprints_path = 'E:/CPI 빅데이터/시너지_서현수집/compounds_ob-fps.csv'
model_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/best_model.pth'

# 데이터 로드 및 준비
largest_subgraph, drug_target_proteins = load_data(network_path, drug_protein_path)
drug_combinations_data = pd.read_csv(drug_combinations_path)
fingerprint_data = pd.read_csv(drug_fingerprints_path)
fingerprint_columns = ['X%d' % i for i in range(1, 167)]  # Assuming 166 fingerprint dimensions

# Create a dictionary of drug fingerprints
drug_fingerprints = {row['drug_db']: row[fingerprint_columns].values.astype(float) for _, row in fingerprint_data.iterrows()}

# Compute PageRank for the largest subgraph
pagerank = nx.pagerank(largest_subgraph)

# 새로운 cell 데이터 로드
new_cell_data = drug_combinations_data[drug_combinations_data['cell'] == '3T3L1']

# 데이터 분할
if len(new_cell_data) >= 7:
    support_set = new_cell_data.sample(frac=0.7, random_state=42)
    test_set = new_cell_data.drop(support_set.index)
else:
    support_set = new_cell_data
    test_set = pd.DataFrame()  # 빈 데이터프레임

print(test_set.columns)


# 데이터셋 준비
def create_dataset(pairs_df):
    pairs_list = [(row1.to_dict(), row2.to_dict()) for i, row1 in pairs_df.iterrows() for j, row2 in pairs_df.iterrows() if i != j]
    dataset = DrugCombinationDataset(pairs_list, pagerank, drug_fingerprints, len(fingerprint_columns), largest_subgraph, drug_target_proteins)
    return dataset

support_set_dataset = create_dataset(support_set)
support_loader = DataLoader(support_set_dataset, batch_size=len(support_set_dataset), shuffle=False)

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = len(largest_subgraph.nodes)
fingerprint_size = len(fingerprint_columns)
hidden_dim = 128  # Assuming same hidden dimension as during training

model = SNN(input_size, fingerprint_size, hidden_dim)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Test set에 대한 예측 수행
def predict_with_model(model, support_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for pair_features, _ in support_loader:
            outputs = model(pair_features.to(device))
            predictions.extend(outputs.cpu().numpy())
    return predictions

# Test set에 대한 예측 수행
test_set_dataset = create_dataset(test_set)
test_loader = DataLoader(test_set_dataset, batch_size=len(test_set_dataset), shuffle=False)

predictions = predict_with_model(model, test_loader)

# 결과 저장
output_file_test = 'E:/CPI 빅데이터/시너지_서현수집/test_set_results.txt'
with open(output_file_test, 'w') as f:
    for idx, (drug1_db, drug2_db, prediction) in enumerate(
            zip(test_set['drug1_db'], test_set['drug2_db'], predictions)):
        # 배열에서 첫 번째 값을 추출
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]
        f.write(f'{drug1_db}_{drug2_db}\t{prediction:.4f}\n')

print(f"Evaluation results for test set saved in {output_file_test}")

