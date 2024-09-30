import torch
import torch.optim as optim
import logging
import os
from preprocess2 import prepare_data, prepare_support_set, DrugCombinationDataset, load_data
from model_MLP import SNN, train_model, evaluate_model

# Logging 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 경로를 여러분의 데이터셋 파일 경로로 설정하세요.
network_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/protein-protein_network.csv'
drug_protein_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/drug_protein.csv'
drug_combinations_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/drug_combinations2.csv'
drug_fingerprints_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/drug-fps.csv'
model_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/best_model.pth'
results_path = 'E:/CPI 빅데이터/DeepTraSynergy-main/DeepTraSynergy-main/data/DrugCombDB/results'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터 설정
batch_size = 32
shuffle = True
num_epochs = 2
learning_rate = 0.001
attention_dim = 128
hidden_size = 64

# Prepare train_loader, largest_subgraph, fingerprint_size, pagerank, drug_fingerprints
train_loader, largest_subgraph, fingerprint_size, pagerank, drug_fingerprints = prepare_data(
    network_path, drug_protein_path, drug_combinations_path, drug_fingerprints_path, batch_size, shuffle)

# Load drug_target_proteins from load_data
_, drug_target_proteins = load_data(network_path, drug_protein_path)

input_size = len(largest_subgraph.nodes)

# 모델 초기화 및 학습
model = SNN(input_size, fingerprint_size, attention_dim)
model.to(device)

criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
logging.info("Training model...")
train_model(model, train_loader, criterion, optimizer, num_epochs, model_save_path=model_path)

# 셀 라인별 평가
logging.info("Evaluating model...")
support_sets, test_dataset = prepare_support_set(drug_combinations_path)

# 결과 저장 디렉토리 생성
os.makedirs(results_path, exist_ok=True)

for cell, support_set in support_sets.items():
    # 각 셀 라인에 대해 support set을 데이터셋으로 변환
    cell_pairs_list = [(pair1.to_dict(), pair2.to_dict()) for i, pair1 in support_set.iterrows() for j, pair2 in
                       support_set.iterrows() if i != j]
    cell_dataset = DrugCombinationDataset(cell_pairs_list, pagerank, drug_fingerprints, fingerprint_size,
                                          largest_subgraph, drug_target_proteins)

    # DataLoader 생성
    cell_loader = torch.utils.data.DataLoader(cell_dataset, batch_size=batch_size, shuffle=False)

    # 평가 및 결과 저장
    output_file = os.path.join(results_path, f"{cell}_results.txt")
    evaluate_model(model, cell_loader, output_file)
