import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from preprocess2 import DrugCombinationDataset, load_data, prepare_data
from model_MLP import SNN
import networkx as nx
import itertools

# 경로 설정
network_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/protein-protein_network.csv'
drug_protein_path = 'E:/CPI 빅데이터/시너지_서현수집/merged_OB_com_protein.csv'
drug_combinations_path = 'E:/CPI 빅데이터/시너지_서현수집/OB_combination.csv'
drug_fingerprints_path = 'E:/CPI 빅데이터/시너지_서현수집/compounds_ob-fps.csv'
pred_drug_fingerprints_path = 'E:/CPI 빅데이터/시너지_서현수집/compounds_lab-fp2.csv'
model_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/best_model.pth'

# Support set 준비
def prepare_support_set():
    train_loader, largest_subgraph, fingerprint_size, pagerank, drug_fingerprints = prepare_data(
        network_path, drug_protein_path, drug_combinations_path, drug_fingerprints_path)
    return train_loader, largest_subgraph, fingerprint_size, pagerank, drug_fingerprints

# Query set 준비
def prepare_query_set(pred_drug_fingerprints_path, largest_subgraph, pagerank, drug_target_proteins):
    pred_fingerprint_data = pd.read_csv(pred_drug_fingerprints_path)
    fingerprint_columns = ['X%d' % i for i in range(1, 167)]

    # 모든 가능한 약물 쌍 생성
    drugs = pred_fingerprint_data['drug_db'].tolist()
    drug_pairs = list(itertools.combinations(drugs, 2))

    # 약물 쌍 데이터프레임 생성
    query_combinations = pd.DataFrame(drug_pairs, columns=['drug1_db', 'drug2_db'])
    query_combinations['cell'] = 'unknown'  # 세포 정보가 없으므로 'unknown'으로 설정
    query_combinations['synergy_score'] = 0  # 실제 시너지 점수를 모르므로 0으로 설정

    # 약물 지문 딕셔너리 생성
    query_drug_fingerprints = {row['drug_db']: row[fingerprint_columns].values.astype(float)
                               for _, row in pred_fingerprint_data.iterrows()}

    # DrugCombinationDataset 생성
    query_pairs_list = [(row.to_dict(), row.to_dict()) for _, row in query_combinations.iterrows()]
    query_dataset = DrugCombinationDataset(query_pairs_list, pagerank, query_drug_fingerprints,
                                           len(fingerprint_columns), largest_subgraph, drug_target_proteins)

    return DataLoader(query_dataset, batch_size=32, shuffle=False), query_combinations

# 예측 함수
def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in loader:
            outputs = model(features.to(device))
            predictions.extend(outputs.cpu().numpy())
    return predictions

# 메인 실행 부분
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Support set 준비
    train_loader, largest_subgraph, fingerprint_size, pagerank, drug_fingerprints = prepare_support_set()

    # 모델 로드
    input_size = len(largest_subgraph.nodes)
    hidden_dim = 128
    model = SNN(input_size, fingerprint_size, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Query set 준비
    _, drug_target_proteins = load_data(network_path, drug_protein_path)
    query_loader, query_combinations = prepare_query_set(pred_drug_fingerprints_path, largest_subgraph, pagerank, drug_target_proteins)

    # 예측 수행
    predictions = predict(model, query_loader, device)

    # 결과 저장
    query_combinations['predicted_synergy'] = predictions
    query_combinations.to_csv('E:/CPI 빅데이터/시너지_서현수집/predicted_synergies2.csv', index=False)

    # 각 query data마다 가장 높은 예측값 선별
    best_predictions = query_combinations.groupby(['drug1_db', 'drug2_db'])['predicted_synergy'].max().reset_index()
    best_predictions.to_csv('E:/CPI 빅데이터/시너지_서현수집/best_predicted_synergies2.csv', index=False)

    print("예측이 완료되었습니다. 결과는 'predicted_synergies.csv'와 'best_predicted_synergies.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main()