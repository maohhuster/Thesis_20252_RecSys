.PHONY: setup test lint run dts-v1 dts-v1b v2-cache v2-router v2-1-router v2-2-router v2-2-segments v2-3-bootstrap v2-4-oracle v2-5-loso v3-router v3-1-router v4-segment-blend cross-domain-steam-readiness clean

setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

run:
	python3 scripts/run_experiment.py --config configs/default.json

dts-v1:
	python3 scripts/analyze_dts_v1.py --config configs/dts_v1.json

dts-v1b:
	python3 scripts/run_dts_v1b_routing.py --config configs/dts_v1b.json

v2-cache:
	python3 scripts/export_v2_candidate_cache.py --config configs/v2_candidate_cache.json

v2-router:
	python3 scripts/train_v2_router.py --config configs/v2_router.json

v2-1-router:
	python3 scripts/run_v2_1_refined_router.py --config configs/v2_1_router.json

v2-2-router:
	python3 scripts/run_v2_2_global_blend.py --config configs/v2_2_router.json

v2-2-segments:
	python3 scripts/analyze_v2_2_segments.py --config configs/v2_2_segment_analysis.json

v2-3-bootstrap:
	python3 scripts/run_v2_3_bootstrap_significance.py --config configs/v2_3_bootstrap_significance.json

v2-4-oracle:
	python3 scripts/run_v2_4_oracle_gap.py --config configs/v2_4_oracle_gap.json

v2-5-loso:
	python3 scripts/run_v2_5_loso_global_blend.py --config configs/v2_5_loso_global_blend.json

v3-router:
	python3 scripts/run_v3_action_router.py --config configs/v3_action_router.json

v3-1-router:
	python3 scripts/run_v3_1_pairwise_gain_router.py --config configs/v3_1_pairwise_gain_router.json

v4-segment-blend:
	python3 scripts/run_v4_segment_level_blend.py --config configs/v4_segment_level_blend.json

cross-domain-steam-readiness:
	python3 scripts/inspect_cross_domain_steam.py --config configs/cross_domain_steam_readiness.json

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
