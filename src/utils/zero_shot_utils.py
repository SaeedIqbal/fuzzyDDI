import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from .fuzzy_ddi import FuzzyDDI
from .temporal_encoder import TemporalEncoder   
from .mc_dropout import MCDropout
#         sim1 = drug_sim_matrix[unseen_drug1].unsqueeze(0)  # [1 x N]
#         sim2 = drug_sim_matrix[unseen_drug2].unsqueeze(0)  # [1 x N]  
#         sim1 = sim1 / sim1.sum(dim=1, keepdim=True)  # Normalize
#         sim2 = sim2 / sim2.sum(dim=1, keepdim=True)  # Normalize
#         sim1 = sim1.view(-1, 1)  # [N x 1]
#         sim2 = sim2.view(-1, 1)  # [N x 1]
def build_similarity_matrix(model):
    embs = model.entity_embeddings.weight.data
    sim = torch.mm(embs, embs.T)
    sim = F.softmax(sim, dim=1)
    return sim

# Usage in inference
sim_matrix = build_similarity_matrix(model)
prob = model.zero_shot_predict(unseen_drug1_idx, unseen_drug2_idx, sim_matrix)
print(f"Zero-shot DDI Probability: {prob.item():.4f}")