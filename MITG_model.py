import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SeqNextDataset(Dataset):
    def __init__(self, df, item_embeddings_dict: Dict[int, np.ndarray],
                 text_embeddings_dict: Optional[Dict[int, np.ndarray]] = None):
        self.df = df.reset_index(drop=True)
        self.item_emb = item_embeddings_dict
        self.text_emb = text_embeddings_dict

        any_item = next(iter(item_embeddings_dict.keys()))
        self.dim_id = item_embeddings_dict[any_item].shape[0]
        self.dim_text = 0 if text_embeddings_dict is None else text_embeddings_dict[any_item].shape[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq: List[int] = row['seq']
        nxt: int = row['next']

        seq_id_emb = [self.item_emb[i] for i in seq]
        seq_id_emb = np.stack(seq_id_emb, axis=0)

        if self.text_emb is not None:
            seq_text_emb = [self.text_emb[i] for i in seq]
            seq_text_emb = np.stack(seq_text_emb, axis=0)
        else:
            seq_text_emb = np.zeros((len(seq), 0), dtype=np.float32)

        return {
            "seq_items": np.array(seq, dtype=np.int64),
            "seq_id_emb": seq_id_emb.astype(np.float32),
            "seq_text_emb": seq_text_emb.astype(np.float32),
            "next_item": np.int64(nxt),
        }


def collate_fn(batch):
    max_len = max(len(b["seq_items"]) for b in batch)
    dim_id = batch[0]["seq_id_emb"].shape[-1]
    dim_text = batch[0]["seq_text_emb"].shape[-1]

    B = len(batch)
    seq_items = torch.full((B, max_len), -1, dtype=torch.long)
    id_emb = torch.zeros(B, max_len, dim_id, dtype=torch.float32)
    text_emb = torch.zeros(B, max_len, dim_text, dtype=torch.float32)

    lengths = torch.zeros(B, dtype=torch.long)
    next_item = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        L = len(b["seq_items"])
        lengths[i] = L
        seq_items[i, :L] = torch.from_numpy(b["seq_items"])
        id_emb[i, :L] = torch.from_numpy(b["seq_id_emb"])
        if dim_text > 0:
            text_emb[i, :L] = torch.from_numpy(b["seq_text_emb"])
        next_item[i] = b["next_item"]

    return {
        "seq_items": seq_items,
        "seq_id_emb": id_emb,
        "seq_text_emb": text_emb,
        "lengths": lengths,
        "next_item": next_item,
    }


class ContextEncoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int):
        super().__init__()
        self.gru = nn.GRU(input_size=dim_in, hidden_size=dim_hidden, batch_first=True)

    def forward(self, x, lengths, k: int):
        xs = []
        for i in range(x.size(0)):
            Li = lengths[i].item()
            l0 = max(0, Li - k)
            xs.append(x[i:i+1, l0:Li, :])
        xk = torch.cat(xs, dim=0)
        _, h = self.gru(xk)
        return h.squeeze(0)        


class ViewScorer(nn.Module):
    def __init__(self, dim_src: int, dim_ctx: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_src*2 + dim_ctx, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, e_src: torch.Tensor, e_tgt: torch.Tensor, h: torch.Tensor):
        B, C, D = e_tgt.size()
        h_exp = h.unsqueeze(1).expand(B, C, h.size(-1))     
        src_exp = e_src.unsqueeze(1).expand(B, C, e_src.size(-1))  
        x = torch.cat([src_exp, e_tgt, h_exp], dim=-1)      
        out = self.mlp(x).squeeze(-1)                       
        return out


class AlphaNet(nn.Module):
    def __init__(self, dim_ctx: int, n_views: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_ctx, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_views)
        )

    def forward(self, h: torch.Tensor):
        logits = self.net(h)
        return F.softmax(logits, dim=-1)


class TrainableMITG(nn.Module):
    def __init__(self, dim_id: int, dim_text: int, cooccur_dict: dict,
                 dim_ctx: int = 128, k_ctx: int = 5, temperature_init: float = 1.0):
        super().__init__()
        self.k_ctx = k_ctx
        self.cooccur_dict = cooccur_dict
        self.beta_c = nn.Parameter(torch.tensor(1.0))  

        dim_in = dim_id + (dim_text if dim_text > 0 else 0)
        self.ctx_encoder = ContextEncoder(dim_in, dim_ctx)

        self.scorer_t = ViewScorer(dim_text if dim_text>0 else dim_id, dim_ctx, hidden=128)
        self.scorer_id = ViewScorer(dim_id, dim_ctx, hidden=128)

        self.alpha = AlphaNet(dim_ctx, n_views=3, hidden=64)            
        self.stable_proj = nn.Linear(dim_in, 1, bias=False)

        self.logT_c = nn.Parameter(torch.tensor(math.log(temperature_init)))
        self.logT_t = nn.Parameter(torch.tensor(math.log(temperature_init)))
        self.logT_id = nn.Parameter(torch.tensor(math.log(temperature_init)))

    def _gather_last_k(self, x, lengths):
        last_idx = torch.clamp(lengths-1, min=0)  
        out = []
        for i in range(x.size(0)):
            out.append(x[i, last_idx[i], :].unsqueeze(0))
        return torch.cat(out, dim=0)   

    def forward(self,
                seq_items: torch.Tensor,
                seq_id_emb: torch.Tensor,
                seq_text_emb: torch.Tensor,
                lengths: torch.Tensor,
                cand_ids: torch.Tensor,
                cand_id_emb: torch.Tensor,
                cand_text_emb: torch.Tensor,
                ) -> Tuple[torch.Tensor, dict]:

        if seq_text_emb.size(-1) > 0:
            seq_in = torch.cat([seq_id_emb, seq_text_emb], dim=-1)
        else:
            seq_in = seq_id_emb

        h = self.ctx_encoder(seq_in, lengths, self.k_ctx)

        last_id = self._gather_last_k(seq_id_emb, lengths)
        if seq_text_emb.size(-1) > 0:
            last_text = self._gather_last_k(seq_text_emb, lengths)
        else:
            last_text = last_id  

        # --- 协作视图 ---
        B, C = cand_ids.size()
        co_scores = []
        for i in range(B):
            last_item = int(seq_items[i, lengths[i]-1].item())
            cand_ids_i = cand_ids[i].tolist()
            co_probs = [ self.cooccur_dict.get((last_item, cand), 0.0) for cand in cand_ids_i ]
            co_scores.append(co_probs)
        co_scores = torch.tensor(co_scores, device=cand_id_emb.device, dtype=torch.float32)
        score_c = self.beta_c * co_scores
        P_c = F.softmax(score_c / torch.exp(self.logT_c), dim=-1)

        # --- Text 视图 ---
        score_t = self.scorer_t(last_text, cand_text_emb if cand_text_emb.size(-1)>0 else cand_id_emb, h)
        P_t = F.softmax(score_t / torch.exp(self.logT_t), dim=-1)

        # --- ID 视图 ---
        score_id = self.scorer_id(last_id, cand_id_emb, h)
        P_id = F.softmax(score_id / torch.exp(self.logT_id), dim=-1)

        # --- 融合 ---
        alpha = self.alpha(h)                    
        a_c, a_t, a_id = alpha[:,0:1], alpha[:,1:2], alpha[:,2:3]
        P_mix_no_stable = a_c * P_c + a_t * P_t + a_id * P_id  

        hist_mean = seq_in.mean(dim=1, keepdim=True)          
        stable_scores = self.stable_proj(hist_mean).squeeze(-1)  
        stable_scores = stable_scores.expand_as(P_mix_no_stable)  
        P_stable = F.softmax(stable_scores, dim=-1)

        P_mix = torch.clamp(P_stable + P_mix_no_stable, min=1e-12)
        P_mix = P_mix / P_mix.sum(dim=-1, keepdim=True)

        logs = {
            "alpha": alpha,
            "T": torch.stack([torch.exp(self.logT_c), torch.exp(self.logT_t), torch.exp(self.logT_id)]).detach().cpu().numpy(),
        }
        return P_mix, logs


def build_candidates(seq_items: torch.Tensor,
                     item_embeddings_dict: Dict[int, np.ndarray],
                     text_embeddings_dict: Optional[Dict[int, np.ndarray]],
                     topM: int = 200):
    device = seq_items.device
    all_item_ids = list(item_embeddings_dict.keys())
    all_id_mat = np.stack([item_embeddings_dict[i] for i in all_item_ids], axis=0)
    all_id_t = torch.from_numpy(all_id_mat).to(device)
    N, D = all_id_t.size()

    id_stack = []
    for i in range(seq_items.size(0)):
        idx = seq_items[i][seq_items[i] >= 0].cpu().numpy().tolist()
        if len(idx) == 0:
            idx = [all_item_ids[0]]
        emb = np.stack([item_embeddings_dict[j] for j in idx], axis=0).mean(axis=0)
        id_stack.append(emb)
    hist_mean_id = torch.from_numpy(np.stack(id_stack, axis=0)).to(device)

    scores = hist_mean_id @ all_id_t.T
    topk = torch.topk(scores, k=min(topM, N), dim=-1)
    cand_idx = topk.indices

    cand_ids_list, cand_id_emb_list, cand_text_emb_list = [], [], []
    dim_text = 0 if text_embeddings_dict is None else next(iter(text_embeddings_dict.values())).shape[0]
    for i in range(seq_items.size(0)):
        ids = [all_item_ids[j] for j in cand_idx[i].cpu().tolist()]
        cand_ids_list.append(ids)
        id_emb = np.stack([item_embeddings_dict[j] for j in ids], axis=0)
        cand_id_emb_list.append(id_emb)
        if text_embeddings_dict is not None:
            te = np.stack([text_embeddings_dict[j] for j in ids], axis=0)
        else:
            te = np.zeros((len(ids), dim_text), dtype=np.float32)
        cand_text_emb_list.append(te)

    cand_ids = torch.tensor(cand_ids_list, dtype=torch.long, device=device)
    cand_id_emb = torch.from_numpy(np.stack(cand_id_emb_list, axis=0)).to(device).float()
    cand_text_emb = torch.from_numpy(np.stack(cand_text_emb_list, axis=0)).to(device).float()
    return cand_ids, cand_id_emb, cand_text_emb


def train_epoch(model: TrainableMITG, loader: DataLoader,
                item_emb_dict: Dict[int, np.ndarray],
                text_emb_dict: Optional[Dict[int, np.ndarray]],
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                topM: int = 200):
    model.train()
    ce_loss_avg, acc_avg, n_total = 0.0, 0.0, 0
    for batch in loader:
        seq_items = batch["seq_items"].to(device)
        seq_id_emb = batch["seq_id_emb"].to(device)
        seq_text_emb = batch["seq_text_emb"].to(device)
        lengths = batch["lengths"].to(device)
        y = batch["next_item"].to(device)

        cand_ids, cand_id_emb, cand_text_emb = build_candidates(seq_items, item_emb_dict, text_emb_dict, topM=topM)
        P_mix, _ = model(seq_items, seq_id_emb, seq_text_emb, lengths,
                         cand_ids, cand_id_emb, cand_text_emb)

        mask = torch.zeros_like(P_mix[:, 0], dtype=torch.bool)
        y_pos = torch.zeros_like(P_mix[:, 0], dtype=torch.long)
        for i in range(seq_items.size(0)):
            idx = (cand_ids[i] == y[i]).nonzero(as_tuple=False)
            if idx.numel() > 0:
                mask[i] = True
                y_pos[i] = idx[0, 0]

        if mask.sum() == 0:
            continue

        P_sel = P_mix[mask]
        y_sel = y_pos[mask]
        loss = F.nll_loss(torch.log(P_sel + 1e-12), y_sel)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pred = P_sel.argmax(dim=-1)
            acc = (pred == y_sel).float().mean().item()
            ce_loss_avg += loss.item() * P_sel.size(0)
            acc_avg += acc * P_sel.size(0)
            n_total += P_sel.size(0)

    if n_total == 0:
        return 0.0, 0.0
    return ce_loss_avg / n_total, acc_avg / n_total


@torch.no_grad()
def infer_topk_for_seq(model: TrainableMITG,
                       seq: List[int],
                       item_emb_dict: Dict[int, np.ndarray],
                       text_embeddings_dict: Optional[Dict[int, np.ndarray]] = None,
                       K: int = 2, topM: int = 200,
                       device: Optional[torch.device] = None,
                       forbidden: Optional[set] = None) -> List[int]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forbidden = forbidden or set()

    id_emb = np.stack([item_emb_dict[i] for i in seq], axis=0).astype(np.float32)
    if text_embeddings_dict is not None:
        text_emb = np.stack([text_embeddings_dict[i] for i in seq], axis=0).astype(np.float32)
    else:
        text_emb = np.zeros((len(seq), 0), dtype=np.float32)

    batch = {
        "seq_items": torch.tensor([seq], dtype=torch.long, device=device),
        "seq_id_emb": torch.tensor(id_emb, dtype=torch.float32, device=device).unsqueeze(0),
        "seq_text_emb": torch.tensor(text_emb, dtype=torch.float32, device=device).unsqueeze(0),
        "lengths": torch.tensor([len(seq)], dtype=torch.long, device=device),
    }

    cand_ids, cand_id_emb, cand_text_emb = build_candidates(batch["seq_items"], item_emb_dict, text_embeddings_dict, topM=topM)
    P_mix, _ = model(batch["seq_items"], batch["seq_id_emb"], batch["seq_text_emb"], batch["lengths"],
                     cand_ids, cand_id_emb, cand_text_emb)

    # --- 排序 ---
    scores = P_mix[0]
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_ids = cand_ids[0, sorted_idx].tolist()

    # --- 先过滤再取前K ---
    filtered = [x for x in sorted_ids if x not in set(seq) and x not in forbidden]
    return filtered[:K]







@torch.no_grad()
def infer_all_positions(model: TrainableMITG,
                        seq: List[int],
                        item_emb_dict: Dict[int, np.ndarray],
                        text_emb_dict: Optional[Dict[int, np.ndarray]] = None,
                        K: int = 2, topM: int = 200,
                        device: Optional[torch.device] = None):
    predictions, targets = [], []
    used_items = set()

    for t in range(1, len(seq)):   # 1 到 n-1
        prefix, target = seq[:t], seq[t]
        recs = infer_topk_for_seq(model, prefix,
                                  item_emb_dict, text_emb_dict,
                                  K=K, topM=topM, device=device,
                                  forbidden=used_items)
        predictions.append(recs)
        targets.append(target)
        used_items.update(recs)

    # ✅ 在最后一个位置也预测一次（prefix = 整个序列）
    prefix = seq[:len(seq)]
    recs = infer_topk_for_seq(model, prefix,
                              item_emb_dict, text_emb_dict,
                              K=K, topM=topM, device=device,
                              forbidden=used_items)
    predictions.append(recs)
    targets.append(None)   # 最后没有 ground-truth，就放 None 或 -1

    return predictions, targets


def run_training(train_df, valid_df,
                 item_embeddings_dict: Dict[int, np.ndarray],
                 cooccur_dict: Dict[Tuple[int, int], float],
                 text_embeddings_dict: Optional[Dict[int, np.ndarray]] = None,
                 epochs: int = 10, batch_size: int = 128,
                 k_ctx: int = 5, lr: float = 1e-3,
                 topM: int = 30, num_workers: int = 4,
                 device: Optional[torch.device] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SeqNextDataset(train_df, item_embeddings_dict, text_embeddings_dict)
    valid_ds = SeqNextDataset(valid_df, item_embeddings_dict, text_embeddings_dict)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)

    model = TrainableMITG(dim_id=train_ds.dim_id, dim_text=train_ds.dim_text,
                          cooccur_dict=cooccur_dict, dim_ctx=128, k_ctx=k_ctx).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, item_embeddings_dict, text_embeddings_dict, optim, device, topM=topM)
        va_loss, va_acc = train_epoch(model, valid_loader, item_embeddings_dict, text_embeddings_dict, optim, device, topM=topM)
        print(f"[Ep {ep:02d}] train CE={tr_loss:.4f} acc={tr_acc:.4f} | valid CE={va_loss:.4f} acc={va_acc:.4f}")

    return model






