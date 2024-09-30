import networkx as nx
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# 데이터 로드 함수
def load_data(network_path, drug_protein_path):
    network_data = pd.read_csv(network_path)
    G = nx.from_pandas_edgelist(network_data, 'protein_a', 'protein_b')

    # 각 노드의 차수(degree)를 계산하여 차수가 높은 순서대로 정렬합니다.
    node_degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)

    # 가장 차수가 높은 1000개의 노드를 선택합니다.
    largest_subgraph_nodes = [node for node, degree in node_degrees[:1000]]

    largest_subgraph = G.subgraph(largest_subgraph_nodes)

    drug_protein_data = pd.read_csv(drug_protein_path)
    drug_target_proteins = {}
    for drug, protein in zip(drug_protein_data['drug'], drug_protein_data['protein']):
        if drug in drug_target_proteins:
            drug_target_proteins[drug].append(protein)
        else:
            drug_target_proteins[drug] = [protein]

    return largest_subgraph, drug_target_proteins

# 데이터셋 클래스 정의
class DrugCombinationDataset(Dataset):
    def __init__(self, pairs_list, pagerank, drug_fingerprints, fingerprint_size, largest_subgraph, drug_target_proteins):
        self.pairs_list = pairs_list
        self.pagerank = pagerank
        self.drug_fingerprints = drug_fingerprints
        self.fingerprint_size = fingerprint_size
        self.largest_subgraph = largest_subgraph
        self.drug_target_proteins = drug_target_proteins

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        pair1, pair2 = self.pairs_list[idx]

        # Initialize features with the correct size
        pair_features = torch.zeros(2 * len(self.largest_subgraph.nodes) + 2 * self.fingerprint_size)

        # Assign PageRank values for the first and second drug pairs
        for i, (pair, offset) in enumerate([(pair1, 0), (pair2, len(self.largest_subgraph.nodes))]):
            for drug in ['drug1_db', 'drug2_db']:
                targets = self.drug_target_proteins.get(pair[drug], [])
                for target in targets:
                    if target in self.pagerank:
                        pair_features[offset + list(self.largest_subgraph.nodes).index(target)] = self.pagerank[target]

        # Add fingerprints for the first and second drug pairs
        for i, pair in enumerate([pair1, pair2]):
            fingerprint = self.drug_fingerprints.get(pair['drug1_db'], torch.zeros(self.fingerprint_size))
            pair_features[2 * len(self.largest_subgraph.nodes) + i * self.fingerprint_size: 2 * len(self.largest_subgraph.nodes) + (i + 1) * self.fingerprint_size] = torch.tensor(fingerprint, dtype=torch.float32)

        # Label: Both pairs should have the same label
        label = torch.tensor(pair1['synergy_score'], dtype=torch.float32)

        return pair_features, label

# 데이터 준비 함수
def prepare_data(network_path, drug_protein_path, drug_combinations_path, drug_fingerprints_path, batch_size=32, shuffle=True):
    largest_subgraph, drug_target_proteins = load_data(network_path, drug_protein_path)
    drug_combinations_data = pd.read_csv(drug_combinations_path)
    fingerprint_data = pd.read_csv(drug_fingerprints_path)
    fingerprint_columns = ['X%d' % i for i in range(1, 167)]  # 수정된 부분

    # Create a dictionary of drug fingerprints
    drug_fingerprints = {row['drug_db']: row[fingerprint_columns].values.astype(float) for _, row in fingerprint_data.iterrows()}

    # Compute PageRank for the largest subgraph
    pagerank = nx.pagerank(largest_subgraph)

    # Positive pairs: Filter pairs with synergy=1
    positive_pairs = drug_combinations_data[drug_combinations_data['synergy_score'] == 1]

    # Generate positive pair sets (pair of pairs) for the same cell
    positive_pairs_same_cell = positive_pairs.groupby('cell').filter(lambda x: len(x) > 1)
    positive_pairs_list = []
    for cell, group in positive_pairs_same_cell.groupby('cell'):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if len(positive_pairs_list) >= 1000:
                    break
                pair1 = group.iloc[i].to_dict()
                pair2 = group.iloc[j].to_dict()
                positive_pairs_list.append((pair1, pair2))
            if len(positive_pairs_list) >= 1000:
                break
        if len(positive_pairs_list) >= 1000:
            break

    # Generate negative pairs
    negative_pairs = drug_combinations_data[
        ~drug_combinations_data[['drug1_db', 'drug2_db', 'cell']].apply(tuple, axis=1).isin(
            positive_pairs_same_cell[['drug1_db', 'drug2_db', 'cell']].apply(tuple, axis=1))
    ]
    negative_pairs_list = []
    for i in range(len(negative_pairs)):
        for j in range(i + 1, len(negative_pairs)):
            if len(negative_pairs_list) >= 1000:
                break
            pair1 = negative_pairs.iloc[i].to_dict()
            pair2 = negative_pairs.iloc[j].to_dict()
            negative_pairs_list.append((pair1, pair2))
        if len(negative_pairs_list) >= 1000:
            break

    # Convert to dataset format
    fingerprint_size = len(fingerprint_columns)

    # Create datasets
    positive_dataset = DrugCombinationDataset(positive_pairs_list, pagerank, drug_fingerprints,
                                              fingerprint_size, largest_subgraph, drug_target_proteins)
    negative_dataset = DrugCombinationDataset(negative_pairs_list, pagerank, drug_fingerprints,
                                              fingerprint_size, largest_subgraph, drug_target_proteins)

    # Combine positive and negative datasets
    full_dataset = torch.utils.data.ConcatDataset([positive_dataset, negative_dataset])

    # Create DataLoader instances
    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)

    # Return the necessary information along with the DataLoader
    return data_loader, largest_subgraph, fingerprint_size, pagerank, drug_fingerprints

# Support set 준비 함수
from torch.utils.data import Dataset

def prepare_support_set(drug_combinations_path, max_samples=50):
    drug_combinations_data = pd.read_csv(drug_combinations_path)
    cell_list = drug_combinations_data['cell'].unique()

    support_sets = {}
    test_raw = pd.DataFrame()

    for cell in cell_list:
        cell_data = drug_combinations_data[drug_combinations_data['cell'] == cell]
        positive_pairs = cell_data[cell_data['synergy_score'] == 1]

        if len(positive_pairs) > 0:
            if len(positive_pairs) > max_samples:
                support_set = positive_pairs.sample(max_samples, random_state=42)
            else:
                support_set = positive_pairs
            support_sets[cell] = support_set
            test_raw = pd.concat([test_raw, positive_pairs.drop(support_set.index)], axis=0)

    test_dataset = test_raw.reset_index(drop=True)

    return support_sets, test_dataset



