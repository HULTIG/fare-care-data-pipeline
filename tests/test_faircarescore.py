"""
Tests for FAIR-CARE Score module
"""
import pytest
from faircare.metrics.faircarescore import FAIRCAREScore


@pytest.fixture
def default_config():
    """Default FAIR-CARE config"""
    return {
        "weights": {
            "bronze": 0.3,
            "silver": 0.3,
            "gold": 0.4
        }
    }


def test_faircare_score_initialization(default_config):
    """Test FAIRCAREScore initialization"""
    scorer = FAIRCAREScore(default_config)
    assert scorer.config == default_config


def test_calculate_excellent_score(default_config):
    """Test calculation with excellent scores"""
    scorer = FAIRCAREScore(default_config)
    result = scorer.calculate(sb=0.9, ss=0.9, sg=0.9)
    
    assert result["score"] == 0.9
    assert result["status"] == "EXCELLENT"
    assert result["components"]["bronze"] == 0.9
    assert result["components"]["silver"] == 0.9
    assert result["components"]["gold"] == 0.9


def test_calculate_acceptable_score(default_config):
    """Test calculation with acceptable scores"""
    scorer = FAIRCAREScore(default_config)
    result = scorer.calculate(sb=0.75, ss=0.75, sg=0.75)
    
    assert result["score"] == 0.75
    assert result["status"] == "ACCEPTABLE"


def test_calculate_at_risk_score(default_config):
    """Test calculation with at-risk scores"""
    scorer = FAIRCAREScore(default_config)
    result = scorer.calculate(sb=0.5, ss=0.5, sg=0.5)
    
    assert result["score"] == 0.5
    assert result["status"] == "AT RISK"


def test_weighted_calculation():
    """Test weighted score calculation"""
    config = {
        "weights": {
            "bronze": 0.5,
            "silver": 0.3,
            "gold": 0.2
        }
    }
    scorer = FAIRCAREScore(config)
    result = scorer.calculate(sb=1.0, ss=0.5, sg=0.0)
    
    # Expected: 0.5*1.0 + 0.3*0.5 + 0.2*0.0 = 0.65
    assert abs(result["score"] - 0.65) < 0.01


def test_boundary_excellent_acceptable():
    """Test boundary between EXCELLENT and ACCEPTABLE"""
    config = {"weights": {"bronze": 0.33, "silver": 0.33, "gold": 0.34}}
    scorer = FAIRCAREScore(config)
    
    # Exactly 0.85 should be EXCELLENT
    result = scorer.calculate(sb=0.85, ss=0.85, sg=0.85)
    assert result["status"] == "EXCELLENT"
    
    # Just below 0.85 should be ACCEPTABLE
    result = scorer.calculate(sb=0.84, ss=0.84, sg=0.84)
    assert result["status"] == "ACCEPTABLE"


def test_boundary_acceptable_at_risk():
    """Test boundary between ACCEPTABLE and AT RISK"""
    config = {"weights": {"bronze": 0.33, "silver": 0.33, "gold": 0.34}}
    scorer = FAIRCAREScore(config)
    
    # Exactly 0.70 should be ACCEPTABLE
    result = scorer.calculate(sb=0.70, ss=0.70, sg=0.70)
    assert result["status"] == "ACCEPTABLE"
    
    # Just below 0.70 should be AT RISK
    result = scorer.calculate(sb=0.69, ss=0.69, sg=0.69)
    assert result["status"] == "AT RISK"


def test_zero_scores(default_config):
    """Test with all zero scores"""
    scorer = FAIRCAREScore(default_config)
    result = scorer.calculate(sb=0.0, ss=0.0, sg=0.0)
    
    assert result["score"] == 0.0
    assert result["status"] == "AT RISK"


def test_perfect_scores(default_config):
    """Test with perfect scores"""
    scorer = FAIRCAREScore(default_config)
    result = scorer.calculate(sb=1.0, ss=1.0, sg=1.0)
    
    assert result["score"] == 1.0
    assert result["status"] == "EXCELLENT"


def test_mixed_scores(default_config):
    """Test with mixed layer scores"""
    scorer = FAIRCAREScore(default_config)
    result = scorer.calculate(sb=0.9, ss=0.7, sg=0.8)
    
    # Expected: 0.3*0.9 + 0.3*0.7 + 0.4*0.8 = 0.27 + 0.21 + 0.32 = 0.80
    assert abs(result["score"] - 0.80) < 0.01
    assert result["status"] == "ACCEPTABLE"


def test_components_in_result(default_config):
    """Test that components are included in result"""
    scorer = FAIRCAREScore(default_config)
    result = scorer.calculate(sb=0.8, ss=0.7, sg=0.9)
    
    assert "components" in result
    assert result["components"]["bronze"] == 0.8
    assert result["components"]["silver"] == 0.7
    assert result["components"]["gold"] == 0.9


def test_equal_weights():
    """Test with equal weights"""
    config = {
        "weights": {
            "bronze": 0.33,
            "silver": 0.33,
            "gold": 0.34
        }
    }
    scorer = FAIRCAREScore(config)
    result = scorer.calculate(sb=0.6, ss=0.8, sg=1.0)
    
    # Should be close to average
    expected = (0.6 + 0.8 + 1.0) / 3
    assert abs(result["score"] - expected) < 0.02


def test_gold_heavy_weights():
    """Test with gold-heavy weights"""
    config = {
        "weights": {
            "bronze": 0.1,
            "silver": 0.1,
            "gold": 0.8
        }
    }
    scorer = FAIRCAREScore(config)
    result = scorer.calculate(sb=0.5, ss=0.5, sg=1.0)
    
    # Expected: 0.1*0.5 + 0.1*0.5 + 0.8*1.0 = 0.9
    assert abs(result["score"] - 0.9) < 0.01


def test_missing_weights_uses_defaults():
    """Test that missing weights use defaults"""
    config = {}
    scorer = FAIRCAREScore(config)
    result = scorer.calculate(sb=0.8, ss=0.8, sg=0.8)
    
    # Should use default weights (0.33, 0.33, 0.34 or similar)
    assert 0.7 < result["score"] < 0.9
