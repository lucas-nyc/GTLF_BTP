# Full dataloader with optional GPR feature caching (joblib)
# train/val/test and imputed/missing graph sets.
import os
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from scipy.stats import pearsonr


def _read_csv_no_header(path):
    try:
        df = pd.read_csv(path, header=None)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None
    if df.shape[0] > 0 and df.iloc[0].apply(lambda x: isinstance(x, str)).any():
        df = df.drop(index=0).reset_index(drop=True)
    return df


# --------------------- GPR helpers ---------------------
def extract_node_coords(landmark_coords_df, num_nodes, case_id=None,
                        caseid_col='caseID', landmark_col='Landmark', x_col='X', y_col='Y'):
    """
    Extract per-node coordinates for a given sample (case_id).
    - landmark_coords_df expected to contain rows for many cases with a column that identifies which case a row
      belongs to (caseid_col). For each case_id there should be `num_nodes` rows, one per landmark.
    - If case_id is None, this falls back to the old behavior of taking the first num_nodes rows (kept for safety).
    Returns: (num_nodes, 2) float array with [X, Y] coordinates ordered by landmark index (sorted by landmark_col).
    """
    if landmark_coords_df is None:
        raise ValueError("landmark_coords_df is required to extract node coordinates.")

    # If case_id provided, filter rows for that case
    if case_id is not None:
        df = landmark_coords_df.copy()
        if caseid_col not in df.columns:
            # try lowercase etc
            possible = [c for c in df.columns if c.lower() == caseid_col.lower()]
            if possible:
                caseid_col = possible[0]
        try:
            matched = df[df[caseid_col].astype(int) == int(case_id)]
        except Exception:
            matched = df[df[caseid_col] == case_id]
        if matched.shape[0] == 0:
            raise ValueError(f"No landmark rows found for case_id={case_id} (column '{caseid_col}').")
        # Sort the matched rows by landmark index if available
        if landmark_col in matched.columns:
            try:
                matched_sorted = matched.copy()
                matched_sorted[landmark_col] = matched_sorted[landmark_col].astype(int)
                matched_sorted = matched_sorted.sort_values(by=landmark_col)
            except Exception:
                matched_sorted = matched
        else:
            matched_sorted = matched
        coords = matched_sorted[[x_col, y_col]].to_numpy(dtype=float)
        if coords.shape[0] < num_nodes:
            raise ValueError(f"landmark_coords for case_id={case_id} has {coords.shape[0]} rows but num_nodes={num_nodes} required")
        return coords[:num_nodes].astype(float)

    # fallback: old behavior (take first num_nodes rows)
    arr = np.asarray(landmark_coords_df)
    if arr.shape[0] < num_nodes:
        raise ValueError(f"landmark_coords_df has {arr.shape[0]} rows but num_nodes={num_nodes} required")
    # if arr has more columns than 2, assume [Landmark, X, Y, ...] and pick X,Y
    if arr.shape[1] >= 3:
        return arr[:num_nodes, 1:3].astype(float)
    return arr[:num_nodes].astype(float)


def _normalize_coords_per_case(coords):
    """
    Min-max normalize a (num_nodes,2) coordinate array per case to keep edge attributes
    in a stable range across samples.
    """
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError(f"coords must be (N,2+), got shape={c.shape}")
    c = c[:, :2]
    c_min = np.nanmin(c, axis=0)
    c_max = np.nanmax(c, axis=0)
    span = c_max - c_min
    span = np.where(np.abs(span) < 1e-12, 1.0, span)
    return ((c - c_min) / span).astype(np.float32)


def _build_masked_tabular_row(row_values, mask_node_idx=39, scaler=None, assume_scaled=False, fill_value=-1.0):
    """
    Build the explicit tabular branch input for one sample from the source row.

    The target node is masked after preprocessing to avoid leakage. When a
    scaler is provided, the same fitted scaler used for graph-node temperatures
    is applied here so both views stay numerically aligned.
    """
    row = np.asarray(row_values, dtype=float).reshape(-1).copy()
    row = np.where(np.isnan(row), float(fill_value), row)
    if (not bool(assume_scaled)) and scaler is not None:
        try:
            row = scaler.transform([row])[0]
        except Exception:
            pass
    ti = int(mask_node_idx)
    if 0 <= ti < row.shape[0]:
        row[ti] = float(fill_value)
    return np.asarray(row, dtype=float)


def _build_edge_attr_from_coord(coord, edge_index):
    """
    Edge attributes: [distance, dx, dy] for each directed edge.
    """
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.float)
    coord_t = torch.as_tensor(coord, dtype=torch.float32)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    dx = coord_t[src, 0] - coord_t[dst, 0]
    dy = coord_t[src, 1] - coord_t[dst, 1]
    d = torch.sqrt(dx * dx + dy * dy + 1e-12)
    return torch.stack([d, dx, dy], dim=1).to(dtype=torch.float)


def _attach_spatial_graph_attrs(data, edge_index, case_id, num_nodes, landmark_coords_df, coord_cache):
    """
    Attach `coord` and `edge_attr` to a graph Data object when landmark coordinates
    are available for the given case_id.
    """
    if landmark_coords_df is None or case_id is None:
        return
    if coord_cache is None:
        coord_cache = {}
    try:
        cid = int(case_id)
    except Exception:
        return

    if cid in coord_cache:
        coord_norm = coord_cache[cid]
    else:
        try:
            coords = extract_node_coords(landmark_coords_df, num_nodes=num_nodes, case_id=cid)
            coord_norm = _normalize_coords_per_case(coords)
            coord_cache[cid] = coord_norm
        except Exception as e:
            coord_cache[cid] = None
            warn_cnt = int(coord_cache.get("__warn_count__", 0))
            if warn_cnt < 5:
                print(f"[WARN] Could not attach spatial edge attrs for case_id={cid}: {e}")
            coord_cache["__warn_count__"] = warn_cnt + 1
            return

    if coord_norm is None:
        return
    if int(coord_norm.shape[0]) != int(num_nodes):
        return

    data.coord = torch.tensor(coord_norm[:, :2], dtype=torch.float)
    data.edge_attr = _build_edge_attr_from_coord(data.coord, edge_index)


def _resolve_temperature_feature_idx(cfg, raw_df):
    """
    Resolve the temperature feature index from config or dataframe columns.
    Priority:
      1) cfg.TEMPERATURE_FEATURE_IDX (int)
      2) cfg.TEMPERATURE_COLUMN_NAME (string, case-insensitive exact match)
    Returns:
      int index or None when unresolved.
    """
    idx = getattr(cfg, "TEMPERATURE_FEATURE_IDX", None)
    if idx is not None:
        try:
            idx = int(idx)
            if raw_df is None or (0 <= idx < int(raw_df.shape[1])):
                return idx
        except Exception:
            pass

    col_name = getattr(cfg, "TEMPERATURE_COLUMN_NAME", None)
    if raw_df is not None and col_name is not None:
        try:
            cands = [c for c in raw_df.columns if str(c).strip().lower() == str(col_name).strip().lower()]
            if cands:
                return int(raw_df.columns.get_loc(cands[0]))
        except Exception:
            pass
    return None


def compute_laplacian_positional_encoding(edge_index, num_nodes, pe_dim=8, normalized=True):
    """
    Compute Laplacian positional encodings (LapPE) for a fixed graph topology.
    Returns tensor of shape (num_nodes, pe_dim).
    """
    n = int(num_nodes)
    k = int(max(0, pe_dim))
    if n <= 0 or k <= 0:
        return torch.zeros((max(0, n), max(0, k)), dtype=torch.float)

    A = np.zeros((n, n), dtype=np.float64)
    if edge_index is not None and edge_index.numel() > 0:
        ei = edge_index.detach().cpu().numpy()
        src = ei[0].astype(int, copy=False)
        dst = ei[1].astype(int, copy=False)
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        src = src[valid]
        dst = dst[valid]
        A[src, dst] = 1.0
        A[dst, src] = 1.0

    deg = np.sum(A, axis=1)
    if bool(normalized):
        inv_sqrt = np.zeros_like(deg, dtype=np.float64)
        nz = deg > 1e-12
        inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        D_inv = np.diag(inv_sqrt)
        L = np.eye(n, dtype=np.float64) - (D_inv @ A @ D_inv)
    else:
        L = np.diag(deg) - A

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
    except Exception:
        return torch.zeros((n, k), dtype=torch.float)

    start = 1 if n > 1 else 0  # skip trivial constant mode
    end = min(n, start + k)
    pe = eigvecs[:, start:end]
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])), mode="constant")

    # Deterministic sign convention for eigenvectors.
    for j in range(pe.shape[1]):
        col = pe[:, j]
        if col.size == 0:
            continue
        piv = int(np.argmax(np.abs(col)))
        if col[piv] < 0:
            pe[:, j] = -col

    return torch.tensor(pe.astype(np.float32), dtype=torch.float)


def augment_graph_for_graphsage_egnn(
    data,
    edge_index,
    lap_pe,
    temperature_value=0.0,
    landmark_coords_df=None,
    coord_cache=None,
    append_coord_to_x=True,
    append_temperature_to_x=True,
    append_lappe_to_x=True,
    append_observed_to_x=True
):
    """
    Augment one PyG Data graph with:
      - coord (Nx2)
      - edge_attr [d, dx, dy] if missing
      - temperature node feature (broadcast scalar)
      - LapPE node features
    """
    if data is None:
        return data
    if bool(getattr(data, "_graphsage_egnn_augmented", False)):
        return data

    if not hasattr(data, "x") or data.x is None:
        return data

    x = data.x.to(dtype=torch.float32)
    num_nodes = int(x.shape[0])
    cid = getattr(data, "case_id", None)

    # Ensure coord exists.
    coord = getattr(data, "coord", None)
    if coord is None:
        if landmark_coords_df is not None and cid is not None:
            if coord_cache is None:
                coord_cache = {}
            try:
                icid = int(cid)
                if icid in coord_cache:
                    coord_np = coord_cache[icid]
                else:
                    coord_np = _normalize_coords_per_case(
                        extract_node_coords(landmark_coords_df, num_nodes=num_nodes, case_id=icid)
                    )
                    coord_cache[icid] = coord_np
                if coord_np is not None and int(coord_np.shape[0]) == num_nodes:
                    coord = torch.tensor(coord_np[:, :2], dtype=torch.float)
            except Exception:
                coord = None
        if coord is None:
            coord = torch.zeros((num_nodes, 2), dtype=torch.float)
    else:
        coord = torch.as_tensor(coord, dtype=torch.float)
        if coord.dim() != 2 or coord.shape[0] != num_nodes:
            coord = torch.zeros((num_nodes, 2), dtype=torch.float)
        elif coord.shape[1] > 2:
            coord = coord[:, :2]
        elif coord.shape[1] < 2:
            coord = torch.cat([coord, torch.zeros((num_nodes, 2 - coord.shape[1]), dtype=torch.float)], dim=1)

    data.coord = coord

    if getattr(data, "edge_attr", None) is None:
        data.edge_attr = _build_edge_attr_from_coord(coord, edge_index)

    # Temperature scalar broadcast to nodes.
    tval = float(temperature_value) if np.isfinite(float(temperature_value)) else 0.0
    temp_feat = torch.full((num_nodes, 1), tval, dtype=torch.float)
    data.temperature = temp_feat

    observed = getattr(data, "observed", None)
    if observed is None:
        observed_feat = torch.ones((num_nodes, 1), dtype=torch.float)
    else:
        observed_t = torch.as_tensor(observed, dtype=torch.float).view(-1, 1)
        if observed_t.shape[0] != num_nodes:
            observed_feat = torch.ones((num_nodes, 1), dtype=torch.float)
        else:
            observed_feat = observed_t
    data.observed_feat = observed_feat

    # LapPE tensor.
    if lap_pe is None:
        lap_pe = torch.zeros((num_nodes, 0), dtype=torch.float)
    lap_pe = torch.as_tensor(lap_pe, dtype=torch.float)
    if lap_pe.dim() != 2 or lap_pe.shape[0] != num_nodes:
        lap_pe = torch.zeros((num_nodes, 0), dtype=torch.float)
    data.lap_pe = lap_pe

    blocks = [x]
    if bool(append_coord_to_x):
        blocks.append(coord)
    if bool(append_temperature_to_x):
        blocks.append(temp_feat)
    if bool(append_observed_to_x):
        blocks.append(observed_feat)
    if bool(append_lappe_to_x) and lap_pe.shape[1] > 0:
        blocks.append(lap_pe)

    data.x = torch.cat(blocks, dim=1).to(dtype=torch.float)
    data._graphsage_egnn_augmented = True
    return data



# ----------------- (unchanged) Imputed dataset loaders -----------------
def load_imputed_for_method(method_dir, method_name, test_indices, mnar_regex, mcar_regex, mnar_mode=False):
    """
    Returns records as list of dicts:
      {'set_idx': set_idx, 'scenario': scenario, 'df': subset_df, 'case_ids': case_ids}
   where case_ids is an array of original row indices (in INPUT_DATA) corresponding to df rows.
    """
    records = []
    if not os.path.isdir(method_dir):
        return records

    files = sorted(glob.glob(os.path.join(method_dir, "*.csv")))
    seen = set()
    for fpath in files:
        fname = os.path.basename(fpath)
        if mnar_mode:
            m = re.match(mnar_regex, fname, flags=re.IGNORECASE) if mnar_regex else None
            if not m:
                continue
            set_idx = int(m.group('idx'))
            scenario = m.group('scenario')
        else:
            m = re.match(mcar_regex, fname, flags=re.IGNORECASE) if mcar_regex else None
            if not m:
                continue
            set_idx = int(m.group('idx'))
            scenario = "MCAR"

        key = (set_idx, scenario.lower())
        if key in seen:
            continue
        seen.add(key)

        df = _read_csv_no_header(fpath)
        if df is None:
            continue

        try:
            # subset without dropping original indices â€” we will store the mapping separately
            subset_df = df.iloc[test_indices].copy()
            case_ids = np.asarray(test_indices, dtype=int).tolist()
            # reset index for returned df (user-facing) but keep case_ids for mapping
            subset_df = subset_df.reset_index(drop=True)
        except Exception as e:
            print(f"[WARN] Failed to index test rows for {fname}: {e}")
            continue

        records.append({'set_idx': set_idx, 'scenario': scenario, 'df': subset_df, 'case_ids': case_ids})

    records = sorted(records, key=lambda r: r['set_idx'])
    return records


def load_all_imputed(base_dir, methods, test_indices, mnar_subdir="MNAR", mcar_subdir="MCAR",
                     mnar_regex=None, mcar_regex=None):
    result = {}
    for method in methods:
        mnar_folder = os.path.join(base_dir, mnar_subdir, method)
        recs_mnar = load_imputed_for_method(mnar_folder, method, test_indices, mnar_regex, mcar_regex, mnar_mode=True)
        result[f"MNAR_{method}"] = recs_mnar

        mcar_folder = os.path.join(base_dir, mcar_subdir, method)
        recs_mcar = load_imputed_for_method(mcar_folder, method, test_indices, mnar_regex, mcar_regex, mnar_mode=False)
        result[f"MCAR_{method}"] = recs_mcar

    print(f"[DEBUG][load_all_imputed] loaded imputed keys: {list(result.keys())}")
    for k, v in result.items():
        print(f"  {k}: {len(v)} sets")
    return result


# ----------------- Missing dataset loader -----------------
def load_missing_datasets(directory, test_indices, missing_regex):
    """
    Returns dict mapping scenario_label -> (list_of_DataFrames, list_of_caseids_lists)
    """
    grouped = defaultdict(list)
    if not os.path.isdir(directory):
        return {}

    files = sorted(glob.glob(os.path.join(directory, "missing_dataset*.csv")))
    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.match(missing_regex, fname, flags=re.IGNORECASE) if missing_regex else None
        if not m:
            continue
        set_idx = int(m.group('idx'))
        scenario = m.group('scenario') if m.group('scenario') else None
        if scenario is None:
            if fname.lower().endswith("_percent.csv"):
                scenario_label = f"{set_idx}_percent"
            else:
                scenario_label = "unknown"
        else:
            scenario_label = scenario.lower()

        df = _read_csv_no_header(fpath)
        if df is None:
            continue

        try:
            # preserve mapping to original rows via test_indices
            test_df_raw = df.iloc[test_indices].copy()
            case_ids = np.asarray(test_indices, dtype=int).tolist()
            test_df = test_df_raw.reset_index(drop=True)
        except Exception as e:
            print(f"[WARN] Failed to index test rows for {fname}: {e}")
            continue

        grouped[scenario_label].append((set_idx, test_df, case_ids))

    grouped_sorted = {}
    for scen, lst in grouped.items():
        # produce list of dataframes (for compatibility), but store case_ids separately
        sorted_list = [df for _, df, _ in sorted(lst, key=lambda x: x[0])]
        caseids_list = [case_ids for _, _, case_ids in sorted(lst, key=lambda x: x[0])]
        grouped_sorted[scen] = (sorted_list, caseids_list)


    return grouped_sorted


def create_sets_of_graphs_from_df_list(
    df_list,
    scaler,
    edge_index,
    mask_node_idx=39,
    case_ids_list=None,
    landmark_coords_df=None,
    coord_cache=None
):
    """
    Convert a list of DataFrames (each df is a set) into list-of-graph-sets.
    Each graph Data will contain:
      - data.raw_row: the original (unnormalized) numeric row used to build the graph
     - data.case_id: the original case index in INPUT_DATA (for mapping to LANDMARK_COORDS)
      - data.observed: explicit boolean mask (True if original row had value)
      - x, y (normalized) as before for modeling

    case_ids_list: optional list-of-lists parallel to df_list where each entry is the list of
                   original case IDs corresponding to that DataFrame's rows.
    """
    all_sets = []
    if coord_cache is None:
        coord_cache = {}
    if case_ids_list is None:
        case_ids_list = [None] * len(df_list)

    for df, case_ids in zip(df_list, case_ids_list):
        df_num = df.select_dtypes(include=[np.number]).astype(float).copy()
        set_graphs = []
        # if case_ids provided, it should match df_num.shape[0]
        if case_ids is not None and len(case_ids) != df_num.shape[0]:
            # attempt to broadcast or fallback
            case_ids = None
        for rid in range(df_num.shape[0]):
            # keep original NaN mask for this row (RAW units)
            row_orig = df_num.iloc[rid].to_numpy(dtype=float, copy=True)
            observed_mask = ~np.isnan(row_orig)
            norm_row = _build_masked_tabular_row(
                row_orig,
                mask_node_idx=mask_node_idx,
                scaler=scaler,
                assume_scaled=False,
                fill_value=-1.0,
            )
            if 0 <= mask_node_idx < norm_row.shape[0]:
                observed_mask[mask_node_idx] = False

            x_tensor = torch.tensor(norm_row.reshape(-1, 1), dtype=torch.float)
            y_tensor = torch.tensor(norm_row.reshape(-1, 1), dtype=torch.float)

            data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
            # store observed mask explicitly (boolean)
            data.observed = torch.tensor(observed_mask.astype(bool), dtype=torch.bool)
            # explicit tabular branch view built from the same source row
            data.tabular = torch.tensor(norm_row.reshape(1, -1), dtype=torch.float)

            # attach raw row (unnormalized) and original case_id (if provided)
            data.raw_row = np.array(row_orig, dtype=float)
            data.case_id = int(case_ids[rid]) if case_ids is not None else None
            _attach_spatial_graph_attrs(
                data=data,
                edge_index=edge_index,
                case_id=data.case_id,
                num_nodes=int(norm_row.shape[0]),
                landmark_coords_df=landmark_coords_df,
                coord_cache=coord_cache
            )

            set_graphs.append(data)
        all_sets.append(set_graphs)

    try:
        print(f"[DEBUG][create_sets_of_graphs_from_df_list] created {len(all_sets)} sets; sample sizes: {[len(s) for s in all_sets][:6]}")
        if len(all_sets) > 0 and len(all_sets[0]) > 0:
            g0 = all_sets[0][0]
            x0 = g0.x.cpu().numpy()
            y0 = g0.y.cpu().numpy()
            print(f"[DEBUG] sample set graph: x.shape={x0.shape}, x.min={np.nanmin(x0):.6g}, x.max={np.nanmax(x0):.6g}, y.shape={y0.shape}, y[0:5]={y0.flatten()[0:5]}")
    except Exception:
        pass

    return all_sets


def compute_edges(subset_data, threshold=0.6):
    num_nodes = subset_data.shape[1]
    corr = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            try:
                rho, _ = pearsonr(subset_data[:, i], subset_data[:, j])
            except Exception:
                rho = 0.0
            corr[i, j] = corr[j, i] = rho
    edge_list = [[i, j] for i in range(num_nodes) for j in range(i + 1, num_nodes) if abs(corr[i, j]) >= threshold]
    edge_list += [[j, i] for i, j in edge_list]
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def create_graphs(
    data_normalized,
    edge_index,
    mask_node_idx=39,
    case_ids=None,
    raw_rows=None,
    scaler=None,
    landmark_coords_df=None,
    coord_cache=None
):
    graphs = []
    if coord_cache is None:
        coord_cache = {}
    n_samples, num_nodes = data_normalized.shape
    if case_ids is not None:
        try:
            case_ids = np.asarray(case_ids).astype(int)
            if case_ids.shape[0] != n_samples:
                case_ids = None
        except Exception:
            case_ids = None
    if raw_rows is not None:
        try:
            raw_rows = np.asarray(raw_rows, dtype=float)
            if raw_rows.shape != data_normalized.shape:
                raw_rows = None
        except Exception:
            raw_rows = None
    for i in range(n_samples):
        x = data_normalized[i].copy()
        y = data_normalized[i].copy()
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
        x_masked = x.copy()
        x_masked[mask_node_idx] = -1.0
        raw_row = raw_rows[i].copy() if raw_rows is not None else None
        tabular_row = _build_masked_tabular_row(
            raw_row if raw_row is not None else x,
            mask_node_idx=mask_node_idx,
            scaler=scaler if raw_row is not None else None,
            assume_scaled=raw_row is None,
            fill_value=-1.0,
        )
        x_tensor = torch.tensor(x_masked.reshape(-1, 1), dtype=torch.float)
        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, num_nodes=num_nodes,
                    num_edges=edge_index.shape[1] if edge_index.numel() else 0)
        # full graphs: all nodes observed except masked target
        observed_mask = np.ones(num_nodes, dtype=bool)
        if 0 <= mask_node_idx < num_nodes:
            observed_mask[mask_node_idx] = False
        data.observed = torch.tensor(observed_mask, dtype=torch.bool)
        data.tabular = torch.tensor(tabular_row.reshape(1, -1), dtype=torch.float)
        if raw_row is not None:
            data.raw_row = np.array(raw_row, dtype=float)
        data.case_id = int(case_ids[i]) if case_ids is not None else None
        _attach_spatial_graph_attrs(
            data=data,
            edge_index=edge_index,
            case_id=data.case_id,
            num_nodes=int(num_nodes),
            landmark_coords_df=landmark_coords_df,
            coord_cache=coord_cache
        )
        graphs.append(data)
    if len(graphs) > 0:
        print(f"[DEBUG][create_graphs] created {len(graphs)} graphs; node_count={graphs[0].num_nodes}, feature_dim={graphs[0].x.shape[1]}")
    return graphs


def load_dataset(cfg):
    """
    Main loader (PCC-only graph construction, train/val/test split).
    Returns:
      (
        train_graphs, test_graphs,
        scaler, edge_index,
        imputed_records, imputed_graph_sets,
        missing_records, missing_graph_sets,
        metadata
      )
    The explicit validation split is stored in metadata for backward-compatible
    callers that still expect the 9-item return signature.
   """
    data_df = pd.read_csv(str(cfg.INPUT_DATA), header=0)
    landmark_coords_df = pd.read_csv(str(cfg.LANDMARK_COORDS), header=0) if getattr(cfg, "LANDMARK_COORDS", None) else None

    num_samples = len(data_df)
    indices = np.arange(num_samples)
    np.random.seed(getattr(cfg, "RANDOM_SEED", 42))
    np.random.shuffle(indices)

    train_size_cfg = getattr(cfg, "TRAIN_SIZE", None)
    if train_size_cfg is None:
        train_ratio = float(getattr(cfg, "TRAIN_RATIO", 0.6))
        val_ratio = float(getattr(cfg, "VAL_RATIO", 0.2))
        test_ratio = float(getattr(cfg, "TEST_RATIO", max(0.0, 1.0 - train_ratio - val_ratio)))
        if not np.isfinite(train_ratio):
            train_ratio = 0.6
        if not np.isfinite(val_ratio):
            val_ratio = 0.2
        if not np.isfinite(test_ratio):
            test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
        total_ratio = float(train_ratio + val_ratio + test_ratio)
        if total_ratio <= 0.0:
            train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
            total_ratio = 1.0
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        allow_zero_val = bool(val_ratio <= 1e-12)

        train_size = int(np.floor((num_samples * train_ratio) + 0.5))
        val_size = int(np.floor((num_samples * val_ratio) + 0.5))
        test_size = num_samples - train_size - val_size

        min_required = 2 if allow_zero_val and num_samples >= 2 else (3 if num_samples >= 3 else num_samples)
        if num_samples >= min_required:
            train_size = max(1, train_size)
            val_size = 0 if allow_zero_val else max(1, val_size)
            test_size = max(1, test_size)
            total_now = train_size + val_size + test_size
            while total_now > num_samples:
                if train_size >= val_size and train_size >= test_size and train_size > 1:
                    train_size -= 1
                elif (not allow_zero_val) and val_size >= test_size and val_size > 1:
                    val_size -= 1
                elif test_size > 1:
                    test_size -= 1
                total_now = train_size + val_size + test_size
            while total_now < num_samples:
                train_size += 1
                total_now = train_size + val_size + test_size
        else:
            train_size = max(1, min(train_size, num_samples))
            if allow_zero_val:
                val_size = 0
            else:
                val_size = max(0, min(val_size, num_samples - train_size))
            test_size = max(0, num_samples - train_size - val_size)
    else:
        train_size = int(train_size_cfg)
        val_ratio = float(getattr(cfg, "VAL_RATIO", 0.2))
        allow_zero_val = bool(val_ratio <= 1e-12)
        min_remaining = 1 if allow_zero_val else 2
        train_size = max(1, min(train_size, num_samples - min_remaining))
        remaining = max(min_remaining, num_samples - train_size)
        val_ratio = float(getattr(cfg, "VAL_RATIO", 0.2))
        test_ratio = float(getattr(cfg, "TEST_RATIO", 0.2))
        rem_total = float(max(1e-8, val_ratio + test_ratio))
        val_size = int(np.floor((remaining * (val_ratio / rem_total)) + 0.5))
        if allow_zero_val:
            val_size = 0
        else:
            val_size = max(1, min(val_size, remaining - 1))
        test_size = remaining - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:train_size + val_size + test_size]

    train_arr = data_df.iloc[train_idx].to_numpy(dtype=float)
    val_arr = data_df.iloc[val_idx].to_numpy(dtype=float) if len(val_idx) > 0 else np.empty((0, data_df.shape[1]), dtype=float)
    test_arr = data_df.iloc[test_idx].to_numpy(dtype=float)
    print(f"[DEBUG] Dataset sizes: train={train_arr.shape}, val={val_arr.shape}, test={test_arr.shape}")

    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train_arr)
    val_norm = scaler.transform(val_arr) if val_arr.shape[0] > 0 else np.empty((0, train_arr.shape[1]), dtype=float)
    test_norm = scaler.transform(test_arr)
    print(f"[DEBUG] scaler fitted. train_norm range: [{np.nanmin(train_norm):.6g}, {np.nanmax(train_norm):.6g}]")

    edge_index = compute_edges(train_norm, threshold=getattr(cfg, "THRESHOLD", 0.6))
    print(f"[DEBUG] edge_index shape: {edge_index.shape if hasattr(edge_index, 'shape') else None}")

    coord_cache = {}
    train_graphs = create_graphs(
        train_norm, edge_index,
        mask_node_idx=getattr(cfg, "TARGET", 39),
        case_ids=train_idx,
        raw_rows=train_arr,
        scaler=scaler,
        landmark_coords_df=landmark_coords_df,
        coord_cache=coord_cache
    )
    val_graphs = create_graphs(
        val_norm, edge_index,
        mask_node_idx=getattr(cfg, "TARGET", 39),
        case_ids=val_idx,
        raw_rows=val_arr,
        scaler=scaler,
        landmark_coords_df=landmark_coords_df,
        coord_cache=coord_cache
    )
    test_graphs = create_graphs(
        test_norm, edge_index,
        mask_node_idx=getattr(cfg, "TARGET", 39),
        case_ids=test_idx,
        raw_rows=test_arr,
        scaler=scaler,
        landmark_coords_df=landmark_coords_df,
        coord_cache=coord_cache
    )

    # load imputed DF-level records
    imputed_base = getattr(cfg, "IMPUTED_ROOT", None)
    imputed_methods = getattr(cfg, "IMPUTATION_METHODS", [])
    mnar_sub = getattr(cfg, "IMPUTED_MNAR_DIR", "MNAR")
    mcar_sub = getattr(cfg, "IMPUTED_MCAR_DIR", "MCAR")

    imputed_records = {}
    if imputed_base:
        imputed_records = load_all_imputed(
            base_dir=imputed_base,
            methods=imputed_methods,
            test_indices=test_idx,
            mnar_subdir=mnar_sub,
            mcar_subdir=mcar_sub,
            mnar_regex=getattr(cfg, "MNAR_IMPUTED_REGEX", None),
            mcar_regex=getattr(cfg, "MCAR_IMPUTED_REGEX", None),
        )

    # Convert DF records into PCC graph sets
    imputed_graph_sets = {}
    for key, recs in imputed_records.items():
        sets_non = []
        for r in recs:
            case_ids_for_df = r.get('case_ids', None)
            gs_list_non = create_sets_of_graphs_from_df_list(
                [r['df']], scaler, edge_index,
                mask_node_idx=getattr(cfg, "TARGET", 39),
                case_ids_list=[case_ids_for_df],
                landmark_coords_df=landmark_coords_df,
                coord_cache=coord_cache
            )
            if gs_list_non:
                sets_non.append(gs_list_non[0])
        imputed_graph_sets[key] = sets_non

    print(f"[DEBUG] imputed_graph_sets keys: {list(imputed_graph_sets.keys())[:12]}")
    for k, v in list(imputed_graph_sets.items())[:6]:
        print(f"  {k}: #sets={len(v)}; sample len={(len(v[0]) if len(v)>0 else 0)}")
    # missing datasets (DF-level)
    missing_base = getattr(cfg, "MISSING_ROOT", None)
    missing_records = {}
    if missing_base:
        missing_mnar_dir = getattr(cfg, "MISSING_MNAR_DIR", "MNAR")
        missing_mcar_dir = getattr(cfg, "MISSING_MCAR_DIR", "MCAR")
        missing_records = {
            "MNAR": load_missing_datasets(os.path.join(missing_base, missing_mnar_dir), test_idx, getattr(cfg, "MISSING_REGEX", None)),
            "MCAR": load_missing_datasets(os.path.join(missing_base, missing_mcar_dir), test_idx, getattr(cfg, "MISSING_REGEX", None))
        }

    # convert missing df lists to PCC graph sets grouped by scenario
    missing_graph_sets = {"MNAR": {}, "MCAR": {}}
    for topk in ["MNAR", "MCAR"]:
        for scen, pair in (missing_records.get(topk, {}) or {}).items():
            df_list, caseids_list = pair if isinstance(pair, tuple) and len(pair) == 2 else (pair, None)
            raw_sets_non = create_sets_of_graphs_from_df_list(
                df_list, scaler, edge_index,
                mask_node_idx=getattr(cfg, "TARGET", 39),
                case_ids_list=caseids_list,
                landmark_coords_df=landmark_coords_df,
                coord_cache=coord_cache
            )
            missing_graph_sets[topk][scen] = raw_sets_non

    print(f"[DEBUG] missing_graph_sets sample keys: {list(missing_graph_sets.get('MNAR', {}).keys())[:8]}")

    metadata = {
        "landmark_coords_df": landmark_coords_df,
        "pca_loadings": None,
        "indices": {"train": train_idx, "val": val_idx, "test": test_idx},
        "raw": {"train": train_arr, "val": val_arr, "test": test_arr},
        "graphs": {"val": val_graphs}
    }

    return (
       train_graphs, test_graphs,
        scaler, edge_index,
        imputed_records, imputed_graph_sets,
        missing_records, missing_graph_sets,
        metadata
    )


def _extract_temperature_for_graph(
    data,
    temperature_idx,
    full_raw_df=None,
    default_value=0.0,
    target_idx=None
):
    """
    Resolve one scalar temperature for a graph.
    - If `temperature_idx` is provided: use that column.
      Prefer normalized graph-space value (data.y) to keep scale aligned with node features.
    - Else: fallback to row mean temperature (excluding target_idx when available).
    """
    fallback_row = None
    if temperature_idx is None:
        raw_row = getattr(data, "raw_row", None)
        if raw_row is not None:
            try:
                fallback_row = np.asarray(raw_row, dtype=float).reshape(-1)
            except Exception:
                fallback_row = None
        if fallback_row is None:
            cid = getattr(data, "case_id", None)
            if full_raw_df is not None and cid is not None:
                try:
                    icid = int(cid)
                    if 0 <= icid < int(full_raw_df.shape[0]):
                        fallback_row = full_raw_df.iloc[icid, :].to_numpy(dtype=float)
                except Exception:
                    fallback_row = None
        if fallback_row is not None and fallback_row.size > 0:
            try:
                vals = np.asarray(fallback_row, dtype=float).copy()
                if target_idx is not None:
                    ti = int(target_idx)
                    if 0 <= ti < int(vals.shape[0]):
                        vals[ti] = np.nan
                v = float(np.nanmean(vals))
                if np.isfinite(v):
                    return v
            except Exception:
                pass
        return float(default_value)

    # Prefer normalized graph-space value when available (align with node feature scale).
    y = getattr(data, "y", None)
    if y is not None:
        try:
            yt = torch.as_tensor(y, dtype=torch.float).view(-1)
            ti = int(temperature_idx)
            if 0 <= ti < int(yt.shape[0]):
                v = float(yt[ti].detach().cpu().numpy())
                if np.isfinite(v):
                    return v
        except Exception:
            pass

    # Fallback to raw_row when present.
    raw_row = getattr(data, "raw_row", None)
    if raw_row is not None:
        try:
            arr = np.asarray(raw_row, dtype=float).reshape(-1)
            if 0 <= int(temperature_idx) < int(arr.shape[0]):
                v = float(arr[int(temperature_idx)])
                if np.isfinite(v):
                    return v
        except Exception:
            pass

    # Fallback to original full dataset via case_id.
    cid = getattr(data, "case_id", None)
    if full_raw_df is not None and cid is not None:
        try:
            icid = int(cid)
            if 0 <= icid < int(full_raw_df.shape[0]):
                v = float(full_raw_df.iloc[icid, int(temperature_idx)])
                if np.isfinite(v):
                    return v
        except Exception:
            pass

    return float(default_value)


def _augment_graph_list_for_graphsage_egnn(
    graphs,
    edge_index,
    lap_pe,
    temperature_idx=None,
    full_raw_df=None,
    landmark_coords_df=None,
    coord_cache=None,
    append_coord_to_x=True,
    append_temperature_to_x=True,
    append_lappe_to_x=True,
    append_observed_to_x=True,
    temp_default=0.0,
    target_idx=None
):
    if not isinstance(graphs, list):
        return
    for g in graphs:
        tval = _extract_temperature_for_graph(
            g,
            temperature_idx=temperature_idx,
            full_raw_df=full_raw_df,
            default_value=temp_default,
            target_idx=target_idx
        )
        augment_graph_for_graphsage_egnn(
            g,
            edge_index=edge_index,
            lap_pe=lap_pe,
            temperature_value=tval,
            landmark_coords_df=landmark_coords_df,
            coord_cache=coord_cache,
            append_coord_to_x=append_coord_to_x,
            append_temperature_to_x=append_temperature_to_x,
            append_lappe_to_x=append_lappe_to_x,
            append_observed_to_x=append_observed_to_x
        )


def load_dataset_graphsage_egnn(cfg):
    """
    Load dataset with additional augmentation for GraphSAGE+EGNN hybrid:
      - coordinates
      - temperature feature
      - Laplacian positional encoding (LapPE)
    Return signature matches load_dataset(cfg).
    """
    outs = load_dataset(cfg)
    (
        train_graphs, test_graphs,
        scaler, edge_index,
        imputed_records, imputed_graph_sets,
        missing_records, missing_graph_sets,
        metadata
    ) = outs

    landmark_coords_df = metadata.get("landmark_coords_df", None)
    try:
        full_raw_df = pd.read_csv(str(cfg.INPUT_DATA), header=0)
    except Exception:
        full_raw_df = None

    temperature_idx = _resolve_temperature_feature_idx(cfg, full_raw_df)
    target_idx = int(getattr(cfg, "TARGET", 39))
    if temperature_idx is None:
        raise ValueError(
            "[graphsage_egnn] Temperature feature is required but not resolved. "
            "Set cfg.TEMPERATURE_FEATURE_IDX (0-based) or cfg.TEMPERATURE_COLUMN_NAME."
        )
    print(f"[INFO][graphsage_egnn] Temperature index resolved: {int(temperature_idx)}")

    pe_dim = int(getattr(cfg, "GRAPH_SAGE_EGNN_LAPPE_DIM", 8))
    pe_norm = bool(getattr(cfg, "GRAPH_SAGE_EGNN_LAPPE_NORMALIZED", True))
    append_coord_to_x = bool(getattr(cfg, "GRAPH_SAGE_EGNN_APPEND_COORD", True))
    append_temperature_to_x = bool(getattr(cfg, "GRAPH_SAGE_EGNN_APPEND_TEMPERATURE", True))
    append_lappe_to_x = bool(getattr(cfg, "GRAPH_SAGE_EGNN_APPEND_LAPPE", True))
    append_observed_to_x = bool(getattr(cfg, "GRAPH_SAGE_EGNN_APPEND_OBSERVED", True))
    temp_default = float(getattr(cfg, "GRAPH_SAGE_EGNN_TEMPERATURE_DEFAULT", 0.0))

    num_nodes = 0
    if isinstance(train_graphs, list) and len(train_graphs) > 0 and hasattr(train_graphs[0], "x") and train_graphs[0].x is not None:
        num_nodes = int(train_graphs[0].x.shape[0])
    elif isinstance(test_graphs, list) and len(test_graphs) > 0 and hasattr(test_graphs[0], "x") and test_graphs[0].x is not None:
        num_nodes = int(test_graphs[0].x.shape[0])

    lap_pe = compute_laplacian_positional_encoding(
        edge_index=edge_index,
        num_nodes=num_nodes,
        pe_dim=pe_dim,
        normalized=pe_norm
    )

    coord_cache = {}

    _augment_graph_list_for_graphsage_egnn(
        train_graphs, edge_index=edge_index, lap_pe=lap_pe,
        temperature_idx=temperature_idx, full_raw_df=full_raw_df,
        landmark_coords_df=landmark_coords_df, coord_cache=coord_cache,
        append_coord_to_x=append_coord_to_x,
        append_temperature_to_x=append_temperature_to_x,
        append_lappe_to_x=append_lappe_to_x,
        append_observed_to_x=append_observed_to_x,
        temp_default=temp_default,
        target_idx=target_idx
    )
    _augment_graph_list_for_graphsage_egnn(
        test_graphs, edge_index=edge_index, lap_pe=lap_pe,
        temperature_idx=temperature_idx, full_raw_df=full_raw_df,
        landmark_coords_df=landmark_coords_df, coord_cache=coord_cache,
        append_coord_to_x=append_coord_to_x,
        append_temperature_to_x=append_temperature_to_x,
        append_lappe_to_x=append_lappe_to_x,
        append_observed_to_x=append_observed_to_x,
        temp_default=temp_default,
        target_idx=target_idx
    )

    for _, sets in (imputed_graph_sets or {}).items():
        for s in (sets or []):
            _augment_graph_list_for_graphsage_egnn(
                s, edge_index=edge_index, lap_pe=lap_pe,
                temperature_idx=temperature_idx, full_raw_df=full_raw_df,
                landmark_coords_df=landmark_coords_df, coord_cache=coord_cache,
                append_coord_to_x=append_coord_to_x,
                append_temperature_to_x=append_temperature_to_x,
                append_lappe_to_x=append_lappe_to_x,
                append_observed_to_x=append_observed_to_x,
                temp_default=temp_default,
                target_idx=target_idx
            )

    for _, scen_map in (missing_graph_sets or {}).items():
        for _, sets in (scen_map or {}).items():
            for s in (sets or []):
                _augment_graph_list_for_graphsage_egnn(
                    s, edge_index=edge_index, lap_pe=lap_pe,
                    temperature_idx=temperature_idx, full_raw_df=full_raw_df,
                    landmark_coords_df=landmark_coords_df, coord_cache=coord_cache,
                    append_coord_to_x=append_coord_to_x,
                    append_temperature_to_x=append_temperature_to_x,
                    append_lappe_to_x=append_lappe_to_x,
                    append_observed_to_x=append_observed_to_x,
                    temp_default=temp_default,
                    target_idx=target_idx
                )

    metadata = dict(metadata or {})
    metadata["graphsage_egnn"] = {
        "temperature_idx": temperature_idx,
        "lappe_dim": int(pe_dim),
        "lappe_normalized": bool(pe_norm),
        "x_augments": {
            "coord": bool(append_coord_to_x),
            "temperature": bool(append_temperature_to_x),
            "observed": bool(append_observed_to_x),
            "lappe": bool(append_lappe_to_x)
        }
    }

    return (
        train_graphs, test_graphs,
        scaler, edge_index,
        imputed_records, imputed_graph_sets,
        missing_records, missing_graph_sets,
        metadata
    )


_BASE_LOAD_DATASET = load_dataset


def _resolve_pca_temperature_path(cfg):
    path = getattr(cfg, "PCA_TEMP_CSV", None)
    if path:
        return str(path)

    input_data = getattr(cfg, "INPUT_DATA", None)
    if input_data:
        return os.path.join(os.path.dirname(str(input_data)), "pca_temp.csv")

    return "pca_temp.csv"


def _load_pca_temperature_table(cfg, expected_rows=None):
    path = _resolve_pca_temperature_path(cfg)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PCA temperature CSV not found: {path}")

    pca_df = pd.read_csv(path, header=0)
    if pca_df is None or pca_df.empty:
        raise ValueError(f"PCA temperature CSV is empty: {path}")

    keep_cols = [c for c in pca_df.columns if not str(c).strip().lower().startswith("unnamed:")]
    if keep_cols:
        pca_df = pca_df.loc[:, keep_cols].copy()

    pca_num = pca_df.apply(pd.to_numeric, errors="coerce")
    pca_num = pca_num.loc[:, ~pca_num.isna().all(axis=0)].copy()
    if pca_num.shape[1] == 0:
        raise ValueError(f"PCA temperature CSV has no usable numeric columns: {path}")

    if expected_rows is not None and int(pca_num.shape[0]) != int(expected_rows):
        raise ValueError(
            f"PCA temperature row count mismatch: expected {int(expected_rows)}, "
            f"got {int(pca_num.shape[0])} from {path}"
        )

    return path, pca_num


def _prepare_pca_temperature_features(cfg, metadata):
    metadata = metadata or {}
    idx_meta = metadata.get("indices", {}) or {}
    train_idx = np.asarray(idx_meta.get("train", []), dtype=int)
    test_idx = np.asarray(idx_meta.get("test", []), dtype=int)
    expected_rows = int(train_idx.size + test_idx.size)

    path, pca_df = _load_pca_temperature_table(cfg, expected_rows=expected_rows if expected_rows > 0 else None)
    pca_all = pca_df.to_numpy(dtype=float, copy=True)

    if train_idx.size == 0:
        raise ValueError("Training indices are required to fit PCA temperature scaling.")

    if np.any(train_idx < 0) or np.any(train_idx >= int(pca_all.shape[0])):
        raise IndexError("Training indices fall outside PCA temperature CSV row bounds.")

    pca_train = pca_all[train_idx]
    train_fill = np.nanmean(pca_train, axis=0)
    train_fill = np.where(np.isfinite(train_fill), train_fill, 0.0)
    pca_all_filled = np.where(np.isfinite(pca_all), pca_all, train_fill.reshape(1, -1))

    pca_scaler = MinMaxScaler()
    pca_scaler.fit(pca_all_filled[train_idx])
    pca_scaled = pca_scaler.transform(pca_all_filled).astype(np.float32, copy=False)

    return {
        "path": path,
        "columns": [str(c) for c in pca_df.columns.tolist()],
        "raw_values": pca_all_filled,
        "scaled_values": pca_scaled,
        "scaler": pca_scaler,
    }


def _append_pca_temperature_to_graph(data, pca_vector):
    if data is None:
        return data
    if bool(getattr(data, "_pca_temperature_augmented", False)):
        return data
    if not hasattr(data, "x") or data.x is None:
        return data

    x = torch.as_tensor(data.x, dtype=torch.float32)
    num_nodes = int(x.shape[0])
    vec = np.asarray(pca_vector, dtype=float).reshape(-1)
    if vec.size == 0:
        return data
    vec = np.where(np.isfinite(vec), vec, 0.0).astype(np.float32, copy=False)

    temp_feat = torch.tensor(
        np.repeat(vec.reshape(1, -1), repeats=num_nodes, axis=0),
        dtype=torch.float
    )
    data.temperature = temp_feat
    data.pca_temperature = temp_feat
    data.pca_temperature_vector = torch.tensor(vec, dtype=torch.float)
    data.x = torch.cat([x, temp_feat], dim=1).to(dtype=torch.float)
    data._pca_temperature_augmented = True
    return data


def _augment_graph_list_with_pca_temperature(graphs, pca_scaled, default_vector=None):
    if not isinstance(graphs, list):
        return

    default_vec = None
    if default_vector is not None:
        default_vec = np.asarray(default_vector, dtype=float).reshape(-1)

    for g in graphs:
        vec = default_vec
        cid = getattr(g, "case_id", None)
        if cid is not None:
            try:
                icid = int(cid)
                if 0 <= icid < int(pca_scaled.shape[0]):
                    vec = pca_scaled[icid]
            except Exception:
                pass
        if vec is None:
            continue
        _append_pca_temperature_to_graph(g, vec)


def load_dataset_pca_temperature(cfg):
    """
    Load the standard dataset pipeline and append per-sample PCA temperature features
    from cfg.PCA_TEMP_CSV to every node in each graph.

    Return signature matches load_dataset(cfg).
    """
    outs = _BASE_LOAD_DATASET(cfg)
    (
        train_graphs, test_graphs,
        scaler, edge_index,
        imputed_records, imputed_graph_sets,
        missing_records, missing_graph_sets,
        metadata
    ) = outs

    pca_info = _prepare_pca_temperature_features(cfg, metadata)
    pca_scaled = pca_info["scaled_values"]
    pca_dim = int(pca_scaled.shape[1]) if pca_scaled.ndim == 2 else 0
    default_vector = np.zeros((pca_dim,), dtype=np.float32)

    _augment_graph_list_with_pca_temperature(
        train_graphs, pca_scaled=pca_scaled, default_vector=default_vector
    )
    _augment_graph_list_with_pca_temperature(
        test_graphs, pca_scaled=pca_scaled, default_vector=default_vector
    )

    for _, sets in (imputed_graph_sets or {}).items():
        for s in (sets or []):
            _augment_graph_list_with_pca_temperature(
                s, pca_scaled=pca_scaled, default_vector=default_vector
            )

    for _, scen_map in (missing_graph_sets or {}).items():
        for _, sets in (scen_map or {}).items():
            for s in (sets or []):
                _augment_graph_list_with_pca_temperature(
                    s, pca_scaled=pca_scaled, default_vector=default_vector
                )

    metadata = dict(metadata or {})
    metadata["pca_temperature"] = {
        "path": pca_info["path"],
        "columns": pca_info["columns"],
        "dim": pca_dim,
        "scaler": pca_info["scaler"],
    }

    if isinstance(train_graphs, list) and len(train_graphs) > 0 and hasattr(train_graphs[0], "x"):
        print(
            f"[INFO][pca_temperature] Loaded {pca_dim} PCA feature(s) from {pca_info['path']} "
            f"-> graph node feature dim={int(train_graphs[0].x.shape[1])}"
        )

    return (
        train_graphs, test_graphs,
        scaler, edge_index,
        imputed_records, imputed_graph_sets,
        missing_records, missing_graph_sets,
        metadata
    )
