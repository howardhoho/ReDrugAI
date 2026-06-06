import pandas as pd
import bigframes.pandas as bpd
from typing import Dict, List, Tuple, Optional
from score import recommend_for_disease_with_similars


def run_disease_evaluation(
    disease_name: str,
    disease_drugs_df,      # DataFrame with disease-drug relationships
    all_drugs_df,          # DataFrame with all available drugs
    moa_pair_sim_df,       # DataFrame with MOA similarity scores
    disease_embedding_df,  # DataFrame with disease embeddings for similarity search
    project_id: str = "redrugai",
    dataset_id: str = "redrugai_data",
    embedding_model_name: str = "embedding005",
    distance_threshold: float = 0.3,
    top_overall: int = 10,
    top_similar: int = 5,
    similar_weight: float = 0.5
) -> Dict:
    """
    Run evaluation experiment for a single disease.
    
    Args:
        disease_name: Name of the disease to evaluate
        disease_drugs_df: DataFrame with known disease-drug relationships
        all_drugs_df: DataFrame with all available drugs
        moa_pair_sim_df: DataFrame with MOA similarity scores
        disease_embedding_df: DataFrame with disease embeddings
        project_id: BigQuery project ID
        dataset_id: BigQuery dataset ID
        embedding_model_name: Name of the embedding model
        distance_threshold: Threshold for similarity search
        top_overall: Number of overall recommendations to return
        top_similar: Number of similar disease recommendations
        similar_weight: Weight for similar disease component
    
    Returns:
        Dictionary with evaluation metrics and results
    """
    
    try:
        # Step 1: Find similar diseases using vector search
        # Escape apostrophes in disease name for SQL safety
        escaped_disease_name = disease_name.replace("'", "''")
        
        vector_search_query = f"""
        WITH query_table AS (
            SELECT *
            FROM ML.GENERATE_EMBEDDING(
                MODEL `{project_id}.{dataset_id}.{embedding_model_name}`,
                (SELECT '{escaped_disease_name}' AS content)
            )
        )
        SELECT
            base.id,
            base.name AS disease_name,
            distance
        FROM
            VECTOR_SEARCH(
                TABLE `{project_id}.{dataset_id}.disease_embedding`,
                'embedding',
                (SELECT * FROM query_table),
                'ml_generate_embedding_result',
                top_k => 100,
                distance_type => 'COSINE'
            )
        WHERE distance < {distance_threshold}
        """
        
        # Execute the similarity search
        similar_disease_df = bpd.read_gbq(vector_search_query)
        # Exclude the query disease itself
        similar_disease_df = similar_disease_df[similar_disease_df['disease_name'] != disease_name]
        similar_disease_names = similar_disease_df['disease_name'].to_list()
        
        # Step 2: Get drug recommendations (with evaluation mode enabled)
        recommendation_result = recommend_for_disease_with_similars(
            disease_name=disease_name,
            similar_disease_names=similar_disease_names,
            disease_drugs_df=disease_drugs_df,
            all_drugs_df=all_drugs_df,
            moa_pair_sim_df=moa_pair_sim_df,
            top_overall=top_overall,
            top_similar=top_similar,
            similar_weight=similar_weight,
            evaluation_mode=True  # Include known drugs for evaluation overlap calculation
        )
        
        # Step 3: Extract results
        known_drugs = recommendation_result["known_drugs"]
        overall_recommendations = recommendation_result["overall_recommendations"]
        similar_recommendations = recommendation_result["similar_recommendations"]
        
        # Step 4: Calculate evaluation metrics
        metrics = calculate_evaluation_metrics(
            known_drugs, overall_recommendations, similar_recommendations
        )
        
        # Step 5: Prepare return data
        result = {
            "disease_name": disease_name,
            "similar_diseases_count": len(similar_disease_names),
            "known_drugs_count": len(known_drugs),
            "overall_recommendations_count": len(overall_recommendations),
            "similar_recommendations_count": len(similar_recommendations),
            "metrics": metrics,
            "success": True,
            "error": None
        }
        
        return result
        
    except Exception as e:
        return {
            "disease_name": disease_name,
            "similar_diseases_count": 0,
            "known_drugs_count": 0,
            "overall_recommendations_count": 0,
            "similar_recommendations_count": 0,
            "metrics": {},
            "success": False,
            "error": str(e)
        }


def calculate_evaluation_metrics(
    known_drugs_df: pd.DataFrame,
    overall_recommendations_df: pd.DataFrame,
    similar_recommendations_df: pd.DataFrame
) -> Dict:
    """
    Calculate evaluation metrics based on overlap between known drugs and recommendations.
    
    Args:
        known_drugs_df: DataFrame with known drugs for the disease
        overall_recommendations_df: DataFrame with overall recommendations
        similar_recommendations_df: DataFrame with similar disease recommendations
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Extract drug sets
    known_drugs_set = set(known_drugs_df['drugId'].astype(str)) if not known_drugs_df.empty else set()
    overall_recs_set = set(overall_recommendations_df['drugId'].astype(str)) if not overall_recommendations_df.empty else set()
    similar_recs_set = set(similar_recommendations_df['drugId'].astype(str)) if not similar_recommendations_df.empty else set()
    
    # Combine all recommendations
    all_recs_set = overall_recs_set.union(similar_recs_set)
    
    # Calculate metrics for overall recommendations
    overall_metrics = calculate_overlap_metrics(known_drugs_set, overall_recs_set, "overall")
    
    # Calculate metrics for similar recommendations
    similar_metrics = calculate_overlap_metrics(known_drugs_set, similar_recs_set, "similar")
    
    # Calculate metrics for combined recommendations
    combined_metrics = calculate_overlap_metrics(known_drugs_set, all_recs_set, "combined")
    
    # Aggregate all metrics
    metrics = {}
    metrics.update(overall_metrics)
    metrics.update(similar_metrics)
    metrics.update(combined_metrics)
    
    return metrics


def calculate_overlap_metrics(known_drugs_set: set, recommendations_set: set, prefix: str) -> Dict:
    """
    Calculate overlap metrics between known drugs and recommendations.
    
    Args:
        known_drugs_set: Set of known drug IDs
        recommendations_set: Set of recommended drug IDs
        prefix: Prefix for metric names
    
    Returns:
        Dictionary with overlap metrics
    """
    
    # Basic counts
    known_count = len(known_drugs_set)
    recs_count = len(recommendations_set)
    
    # Overlap calculations
    overlap = known_drugs_set.intersection(recommendations_set)
    overlap_count = len(overlap)
    
    # Precision: How many recommendations are in known drugs
    # (How many recommendation drugs are in answer group)
    precision = overlap_count / recs_count if recs_count > 0 else 0.0
    
    # Recall: How many known drugs are in recommendations  
    # (How many answer group drugs are in recommendation group)
    recall = overlap_count / known_count if known_count > 0 else 0.0
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        f"{prefix}_known_count": known_count,
        f"{prefix}_recs_count": recs_count,
        f"{prefix}_overlap_count": overlap_count,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1_score": f1_score
    }


def get_random_diseases_sample(
    n_samples: int,
    project_id: str = "redrugai",
    source_project_id: str = "bigquery-public-data",
    source_dataset_id: str = "open_targets_platform",
    min_known_drugs: int = 1
) -> pd.DataFrame:
    """
    Get random sample of diseases from the disease table that have known drugs.
    
    Args:
        n_samples: Number of diseases to sample
        project_id: BigQuery project ID for the target project
        source_project_id: Source project ID for Open Targets data
        source_dataset_id: Source dataset ID for Open Targets data
        min_known_drugs: Minimum number of known drugs required
    
    Returns:
        DataFrame with sampled diseases
    """
    
    query = f"""
    WITH disease_drug_counts AS (
        SELECT 
            d.name as disease_name,
            COUNT(DISTINCT kd.drugId) as known_drug_count
        FROM `{source_project_id}.{source_dataset_id}.disease` d
        INNER JOIN `{source_project_id}.{source_dataset_id}.known_drug` kd
        ON d.id = kd.diseaseId
        GROUP BY d.name
        HAVING COUNT(DISTINCT kd.drugId) >= {min_known_drugs}
    )
    SELECT disease_name, known_drug_count
    FROM disease_drug_counts
    ORDER BY RAND()
    LIMIT {n_samples}
    """
    
    return bpd.read_gbq(query)


if __name__ == "__main__":
    # Example usage
    print("evaluation.py module loaded successfully!")
