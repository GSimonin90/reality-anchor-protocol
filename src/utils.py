import numpy as np

def calculate_bridging_score(post_val, population_beliefs, tolerance=0.25):
    """
    Calculates the bridging score of a post based on its appeal 
    to opposing clusters in the belief distribution.
    
    Ref: Reality Anchor Protocol, Appendix A.
    """
    # Define clusters based on belief distribution (0.0 to 1.0)
    # Left cluster < 0.4, Right cluster > 0.6 
    left_cluster = population_beliefs < 0.4
    right_cluster = population_beliefs > 0.6
    
    # Calculate agreement (likes) within a tolerance window 
    likes = np.abs(population_beliefs - post_val) <= tolerance
    
    # Sum likes from opposing clusters separately
    likes_left = np.sum(likes & left_cluster)
    likes_right = np.sum(likes & right_cluster)
    
    # The score is determined by the MINIMUM support from either side.
    # This penalizes one-sided content (Echo Chambers) and rewards cross-cutting consensus.
    bridging_score = min(likes_left, likes_right)
    
    return bridging_score