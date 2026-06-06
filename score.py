import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional


def normalize_moa(val) -> Optional[str]:
    """Trim + lowercase a MOA string. Return None if null/empty."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    return s if s else None


def make_moa_sim_lookup(moa_sim_df: pd.DataFrame) -> Dict[str, float]:
    """
    Build lookup dict for MOA similarity with columns:
      - 'moa_a' (string)
      - 'moa_b' (string) 
      - 'cosine_similarity' (float)
    Keys are normalized to trim+lower and stored as 'moa_a+moa_b'.
    """
    required = {"moa_a", "moa_b", "cosine_similarity"}
    missing = required.difference(moa_sim_df.columns)
    if missing:
        raise ValueError(f"moa_sim_df is missing columns: {sorted(missing)}")

    pairs = {}
    for r in moa_sim_df.itertuples(index=False):
        moa_a_raw = r.moa_a
        moa_b_raw = r.moa_b
        sim_val = r.cosine_similarity
        if pd.isna(moa_a_raw) or pd.isna(moa_b_raw) or pd.isna(sim_val):
            continue
        
        moa_a_norm = str(moa_a_raw).strip().lower()
        moa_b_norm = str(moa_b_raw).strip().lower()
        if moa_a_norm and moa_b_norm:
            key = f"{moa_a_norm}+{moa_b_norm}"
            pairs[key] = float(sim_val)
    return pairs


def best_moa_similarity(cand_moa: Optional[str], known_moas: List[str], moa_sim: Dict[str, float]) -> float:
    """
    Max similarity across known MOAs using 'moa1+moa2' key.
    Tries both orders: 'cand+known' and 'known+cand'.
    """
    if cand_moa is None or not known_moas:
        return 0.0
    cand = cand_moa.strip().lower()
    best = 0.0
    for km in known_moas:
        km_norm = km.strip().lower()
        best = max(
            best,
            moa_sim.get(f"{cand}+{km_norm}", 0.0),
            moa_sim.get(f"{km_norm}+{cand}", 0.0),
        )
    return best


def score_candidates_against_disease(
    cand_rows,             # DataFrame: rows for candidate drugs (drugId, moa, target_id, ...)
    disease_known_moas,    # list[str] normalized MOAs
    disease_targets,       # set[str] target IDs
    moa_sim_lookup,        # dict key='moa1+moa2' -> similarity
    disease_moa_target_pairs=None,  # set of (moa, target) pairs from known disease drugs
) -> pd.DataFrame:
    """
    Compute per-drug scores of cand_rows **against one disease** using your rules:

      Row score = best_moa_similarity × (2 if row.target ∈ disease_targets else 1)
      Drug sum  = sum(row scores) ; if ANY row uses a disease target => ×10
      Final     = Drug sum / rows_count
    """
    if cand_rows.empty:
        return pd.DataFrame(columns=["drugId", "final_score"])

    # Ensure we're working with pandas DataFrame
    if hasattr(cand_rows, 'to_pandas'):
        rows = cand_rows.to_pandas()
    else:
        rows = cand_rows.copy()
    
    # Handle column name differences
    if "molecule_id" in rows.columns and "drugId" not in rows.columns:
        rows["drugId"] = rows["molecule_id"]
    
    rows["moa_norm"] = rows["moa"].apply(normalize_moa)
    rows["target_id"] = rows["target_id"].astype(str).where(~rows["target_id"].isna(), None)

    rows["_base_sim"] = rows["moa_norm"].apply(lambda m: best_moa_similarity(m, disease_known_moas, moa_sim_lookup))
    rows["_row_has_target"] = rows["target_id"].map(lambda t: (t in disease_targets) if t else False)
    
    # Check for exact MOA-target pair matches
    if disease_moa_target_pairs is None:
        disease_moa_target_pairs = set()
    
    def has_exact_pair_match(moa_norm, target_id):
        if not moa_norm or not target_id:
            return False
        return (moa_norm, str(target_id)) in disease_moa_target_pairs
    
    rows["_has_exact_pair"] = rows.apply(lambda row: has_exact_pair_match(row["moa_norm"], row["target_id"]), axis=1)
    
    # Define multipliers based on match types:
    # - Exact MOA-target pair match: ×5
    # - Target match only: ×2
    # - No target match: ×1  
    def get_multiplier(has_target, has_exact_pair):
        if has_exact_pair:
            return 5.0  # Exact MOA-target pair match
        elif has_target:
            return 2.0  # Target match only
        else:
            return 1.0  # No target match
    
    rows["_multiplier"] = rows.apply(lambda row: get_multiplier(row["_row_has_target"], row["_has_exact_pair"]), axis=1)
    rows["_row_score"] = rows["_base_sim"] * rows["_multiplier"]

    agg = (
        rows.groupby("drugId", as_index=False)
        .agg(
            sum_row_scores=("_row_score", "sum"),
            rows_count=("drugId", "count"),
        )
    )
    # Simple normalization: average score across all rows for this drug
    agg["final_score"] = agg["sum_row_scores"] / agg["rows_count"].replace(0, np.nan)
    return agg[["drugId", "final_score"]]


def prepare_disease_data(disease_name: str, similar_disease_names: List[str], disease_drugs_df, moa_pair_sim_df):
    """
    Normalize inputs and prepare data structures for disease recommendation.
    
    Returns:
        Tuple of (moa_sim_lookup, df, primary_name_norm, similar_names_norm)
    """
    moa_sim_lookup = make_moa_sim_lookup(moa_pair_sim_df.to_pandas())
    
    # Convert BigFrames to pandas for easier processing
    df = disease_drugs_df.to_pandas()
    df["disease_name_norm"] = df["disease_name"].astype(str).str.strip().str.lower()
    primary_name_norm = str(disease_name).strip().lower()
    similar_names_norm = [str(n).strip().lower() for n in (similar_disease_names or [])]
    
    return moa_sim_lookup, df, primary_name_norm, similar_names_norm


def extract_primary_disease_info(df: pd.DataFrame, primary_name_norm: str, disease_name: str):
    """
    Extract primary disease information including drugs, MOAs, targets, and MOA-target pairs.
    
    Returns:
        Dict with keys: has_known_drugs, known_drugs, known_moas, targets, moa_target_pairs, rows
    """
    primary_rows = df[df["disease_name_norm"] == primary_name_norm].copy()
    primary_has_known_drugs = not primary_rows.empty
    
    if primary_has_known_drugs:
        primary_known_drugs = set(primary_rows["drugId"].dropna().astype(str))
        primary_known_moas = [m for m in primary_rows["moa"].apply(normalize_moa) if m is not None]
        primary_targets = set(primary_rows["target_id"].dropna().astype(str))
        
        # Create set of known MOA-target pairs for exact matching
        primary_moa_target_pairs = set()
        for _, row in primary_rows.iterrows():
            moa_norm = normalize_moa(row["moa"])
            target = str(row["target_id"]) if pd.notna(row["target_id"]) else None
            if moa_norm and target:
                primary_moa_target_pairs.add((moa_norm, target))
    else:
        print(f"No known drugs found for {disease_name}. Falling back to similar disease information.")
        primary_known_drugs = set()
        primary_known_moas = []
        primary_targets = set()
        primary_moa_target_pairs = set()
    
    return {
        "has_known_drugs": primary_has_known_drugs,
        "known_drugs": primary_known_drugs,
        "known_moas": primary_known_moas,
        "targets": primary_targets,
        "moa_target_pairs": primary_moa_target_pairs,
        "rows": primary_rows
    }


def extract_similar_diseases_info(df: pd.DataFrame, similar_names_norm: List[str]):
    """
    Extract similar diseases information including drugs and validation.
    
    Returns:
        Dict with keys: rows, known_drugs
    """
    similar_rows = df[df["disease_name_norm"].isin(similar_names_norm)].copy()
    similar_known_drugs = set(similar_rows["drugId"].dropna().astype(str))
    
    return {
        "rows": similar_rows,
        "known_drugs": similar_known_drugs
    }


def score_similar_diseases(similar_rows: pd.DataFrame, candidate_rows: pd.DataFrame, moa_sim_lookup: Dict[str, float]) -> pd.DataFrame:
    """
    Score candidates against each similar disease and return average scores.
    
    Returns:
        DataFrame with columns: drugId, score_similar_mean
    """
    if similar_rows.empty:
        return pd.DataFrame(columns=["drugId", "score_similar_mean"])
    
    # Build per-similar disease info
    sim_groups = similar_rows.groupby("diseaseId")
    sim_score_list = []
    
    for _, grp in sim_groups:
        sim_moas = [m for m in grp["moa"].apply(normalize_moa) if m is not None]
        sim_targets = set(grp["target_id"].dropna().astype(str))
        
        # Create MOA-target pairs for this similar disease
        sim_moa_target_pairs = set()
        for _, row in grp.iterrows():
            moa_norm = normalize_moa(row["moa"])
            target = str(row["target_id"]) if pd.notna(row["target_id"]) else None
            if moa_norm and target:
                sim_moa_target_pairs.add((moa_norm, target))

        s = score_candidates_against_disease(candidate_rows, sim_moas, sim_targets, moa_sim_lookup, sim_moa_target_pairs)
        s = s.rename(columns={"final_score": "score_sim"})
        sim_score_list.append(s)

    if sim_score_list:
        # Average score across similar diseases (outer join then row-wise mean)
        sim_scores = sim_score_list[0]
        for i, s in enumerate(sim_score_list[1:], 1):
            sim_scores = sim_scores.merge(s, on="drugId", how="outer", suffixes=("", f"_{i}"))
        sim_score_cols = [c for c in sim_scores.columns if c.startswith("score_sim")]
        sim_scores["score_similar_mean"] = sim_scores[sim_score_cols].mean(axis=1, skipna=True)
        sim_scores = sim_scores[["drugId", "score_similar_mean"]]
    else:
        sim_scores = pd.DataFrame(columns=["drugId", "score_similar_mean"])
    
    return sim_scores


def get_overall_recommendations(
    all_rows: pd.DataFrame,
    primary_info: Dict,
    similar_info: Dict,
    moa_sim_lookup: Dict[str, float],
    top_overall: int,
    similar_weight: float,
    evaluation_mode: bool = False
) -> pd.DataFrame:
    """
    Get overall recommendations from all candidate drugs.
    
    Args:
        evaluation_mode: If True, include known drugs for evaluation overlap calculation
    
    Returns:
        DataFrame with top overall recommendations
    """
    # Handle column name differences between tables
    if "molecule_id" in all_rows.columns and "drugId" not in all_rows.columns:
        all_rows["drugId"] = all_rows["molecule_id"]
    
    all_rows["drugId"] = all_rows["drugId"].astype(str, errors="ignore")

    if evaluation_mode:
        # For evaluation: include known primary drugs, but exclude similar disease drugs to prevent overlap
        exclude_drugs = similar_info["known_drugs"]
        candidate_rows = all_rows[~all_rows["drugId"].astype(str).isin(exclude_drugs)].copy()
    else:
        # For production: exclude known drugs from both primary and similar diseases
        exclude_drugs = primary_info["known_drugs"].union(similar_info["known_drugs"])
        candidate_rows = all_rows[~all_rows["drugId"].astype(str).isin(exclude_drugs)].copy()

    # Score vs primary (or use similar diseases if no primary drugs)
    if primary_info["has_known_drugs"]:
        scores_primary = score_candidates_against_disease(
            candidate_rows, primary_info["known_moas"], primary_info["targets"], 
            moa_sim_lookup, primary_info["moa_target_pairs"]
        ).rename(columns={"final_score": "score_primary"})
    else:
        # No primary drugs - use similar diseases as primary scoring basis
        scores_primary = pd.DataFrame(columns=["drugId", "score_primary"])
        scores_primary["score_primary"] = 0.0

    # Score vs similar diseases
    sim_scores = score_similar_diseases(similar_info["rows"], candidate_rows, moa_sim_lookup)

    # Combine primary + similar components
    overall = scores_primary.merge(sim_scores, on="drugId", how="left")
    overall["score_similar_mean"] = overall["score_similar_mean"].fillna(0.0)
    
    if primary_info["has_known_drugs"]:
        # Normal case: combine primary and similar scores
        overall["final_score"] = overall["score_primary"] + similar_weight * overall["score_similar_mean"]
    else:
        # Fallback case: use only similar disease scores as the primary basis
        overall["final_score"] = overall["score_similar_mean"]

    # Add drug names to the overall recommendations
    drug_names = all_rows[["drugId", "drug_name"]].drop_duplicates()
    overall = overall.merge(drug_names, on="drugId", how="left")

    overall = (
        overall.sort_values("final_score", ascending=False, kind="mergesort")
        .head(top_overall)
        .reset_index(drop=True)
    )
    
    return overall


def get_similar_recommendations(
    all_rows: pd.DataFrame,
    primary_info: Dict,
    similar_info: Dict,
    moa_sim_lookup: Dict[str, float],
    top_similar: int,
    evaluation_mode: bool = False
) -> pd.DataFrame:
    """
    Get recommendations from similar disease drugs scored against primary disease.
    
    Args:
        evaluation_mode: If True, include known drugs for evaluation overlap calculation
    
    Returns:
        DataFrame with similar disease recommendations
    """
    if evaluation_mode:
        # For evaluation: include all drugs from similar diseases (including primary disease drugs)
        pool_similar_drugs = list(similar_info["known_drugs"])
    else:
        # For production: exclude drugs already known for primary disease
        pool_similar_drugs = list(similar_info["known_drugs"].difference(primary_info["known_drugs"]))
    
    if pool_similar_drugs:
        pool_rows = all_rows[all_rows["drugId"].astype(str).isin(pool_similar_drugs)].copy()
        drug_names = all_rows[["drugId", "drug_name"]].drop_duplicates()
        
        if primary_info["has_known_drugs"]:
            # Normal case: score against primary disease
            sim_recs = score_candidates_against_disease(
                pool_rows, primary_info["known_moas"], primary_info["targets"], 
                moa_sim_lookup, primary_info["moa_target_pairs"]
            ).sort_values("final_score", ascending=False, kind="mergesort")
            
            # Add drug names to similar recommendations
            sim_recs = sim_recs.merge(drug_names, on="drugId", how="left")
            sim_recs = sim_recs.head(top_similar).reset_index(drop=True)
        else:
            # Fallback case: return ALL drugs from similar diseases with basic info
            print(f"Returning all {len(pool_similar_drugs)} drugs from similar diseases.")
            sim_recs = pool_rows[["drugId", "drug_name"]].drop_duplicates().reset_index(drop=True)
            sim_recs["final_score"] = 1.0  # Equal score since we're returning all
    else:
        sim_recs = pd.DataFrame(columns=["drugId", "final_score", "drug_name"])
    
    return sim_recs


def get_known_drugs_display(primary_info: Dict) -> pd.DataFrame:
    """
    Prepare known drugs for display.
    
    Returns:
        DataFrame with known drugs for the primary disease
    """
    if primary_info["has_known_drugs"]:
        primary_rows = primary_info["rows"]
        known_primary_df = (
            primary_rows[["drugId", "drug_name"]].drop_duplicates()
            if "drug_name" in primary_rows.columns
            else primary_rows[["drugId"]].drop_duplicates()
        ).reset_index(drop=True)
    else:
        # No known drugs for primary disease
        known_primary_df = pd.DataFrame(columns=["drugId", "drug_name"])
    
    return known_primary_df


def recommend_for_disease_with_similars(
    disease_name: str,
    similar_disease_names: List[str],
    disease_drugs_df,   # DataFrame with disease-drug relationships: ['diseaseId','disease_name','drugId','moa','target_id']
    all_drugs_df,       # DataFrame with all available drugs: ['molecule_id' or 'drugId','moa','target_id']
    moa_pair_sim_df,    # DataFrame with MOA similarity scores: ['moa_a','moa_b','cosine_similarity']
    top_overall: int = 10,
    top_similar: int = 5,
    similar_weight: float = 0.5, # Weight for the similar-disease component in overall score
    evaluation_mode: bool = False, # Include known drugs for evaluation
) -> Dict[str, pd.DataFrame]:
    """
    Generate drug recommendations for a disease based on similar diseases and mechanism of action (MOA) similarity.
    
    This function implements a drug recommendation system that combines information from:
    1. Known drugs for the primary disease
    2. Known drugs for similar diseases
    3. MOA similarity scores to find candidate drugs
    
    Args:
        disease_name: Primary disease name to find recommendations for
        similar_disease_names: List of similar disease names to use for recommendations
        disease_drugs_df: DataFrame containing disease-drug relationships
        all_drugs_df: DataFrame containing all available drugs with MOA and target information
        moa_pair_sim_df: DataFrame containing pre-computed MOA similarity scores
        top_overall: Number of top overall recommendations to return
        top_similar: Number of top similar disease recommendations to return
        similar_weight: Weight for the similar-disease component in overall score (0.0-1.0)
        
    Returns:
        Dict containing three DataFrames:
        - 'known_drugs': Known drugs for the primary disease
        - 'overall_recommendations': Top-N new drug candidates with combined scoring
        - 'similar_recommendations': Top-M drugs from similar diseases scored against primary
    """

    # Prepare and normalize disease data for processing
    moa_sim_lookup, df, primary_name_norm, similar_names_norm = prepare_disease_data(
        disease_name, similar_disease_names, disease_drugs_df, moa_pair_sim_df
    )

    # Extract information about the primary disease and its known drugs
    primary_info = extract_primary_disease_info(df, primary_name_norm, disease_name)
    
    # Extract information about similar diseases and their known drugs
    similar_info = extract_similar_diseases_info(df, similar_names_norm)

    # Handle edge case: no data available for either primary or similar diseases
    if not primary_info["has_known_drugs"] and similar_info["rows"].empty:
        print(f"No known drugs found for primary disease '{disease_name}' and no similar disease data provided.")
        return {
            "known_drugs": get_known_drugs_display(primary_info),
            "overall_recommendations": pd.DataFrame(columns=["drugId", "drug_name", "final_score"]),
            "similar_recommendations": pd.DataFrame(columns=["drugId", "drug_name", "score_against_primary"]),
        }
    elif not primary_info["has_known_drugs"]:
        print(f"Found {len(similar_info['known_drugs'])} drugs from {len(similar_names_norm)} similar diseases.")

    # Convert BigQuery DataFrame to pandas for efficient processing
    all_rows = all_drugs_df.to_pandas()

    # Generate overall recommendations combining primary and similar disease information
    overall_recommendations = get_overall_recommendations(
        all_rows, primary_info, similar_info, moa_sim_lookup, top_overall, similar_weight, evaluation_mode
    )

    # Generate recommendations specifically from similar diseases
    similar_recommendations = get_similar_recommendations(
        all_rows, primary_info, similar_info, moa_sim_lookup, top_similar, evaluation_mode
    )

    # Prepare known drugs for display
    known_drugs = get_known_drugs_display(primary_info)

    return {
        "known_drugs": known_drugs,
        "overall_recommendations": overall_recommendations[["drugId", "drug_name", "final_score"]],
        "similar_recommendations": similar_recommendations.rename(columns={"final_score": "score_against_primary"}),
    }