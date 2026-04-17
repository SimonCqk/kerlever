"""Tests for the Cross-Candidate Analyzer.

Spec: docs/cross-candidate-analyzer/spec.md
"""

from __future__ import annotations

import json

import pytest

from kerlever.cross_candidate_analyzer import CrossCandidateAnalyzer
from kerlever.types import (
    BenchmarkBundle,
    CandidateIntent,
    CandidateOutcome,
    CompileStatus,
    CorrectnessResult,
    EvaluationResult,
    KernelCandidate,
    Mode,
    ObjectiveScore,
    PerformanceObjective,
    ProblemSpec,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
    SubMode,
)


def _make_spec() -> ProblemSpec:
    """Create a test problem spec."""
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C = A @ B",
        dtype="float16",
        target_gpu="A100",
        shape_cases=[
            ShapeCase(shape_id="small", dims=[128, 128, 128], weight=0.25),
            ShapeCase(shape_id="large", dims=[4096, 4096, 4096], weight=0.75),
        ],
        objective=PerformanceObjective(
            primary_metric="weighted_p50_us",
            aggregation="weighted_mean",
            regression_guard_pct=0.05,
        ),
        target_metric_value=10.0,
        max_rounds=3,
        reference_kernel="__global__ void ref() {}",
    )


def _candidate(
    code_hash: str,
    source_code: str,
    *,
    direction: str = "memory_access",
) -> KernelCandidate:
    """Create a kernel candidate."""
    return KernelCandidate(
        code_hash=code_hash,
        source_code=source_code,
        parent_hashes=["base_hash"],
        intent=CandidateIntent(
            direction=direction,
            mode=Mode.EXPLORE,
            sub_mode=SubMode.RECOMBINATION,
            rationale="test",
        ),
    )


def _result(
    candidate_hash: str,
    outcome: CandidateOutcome,
    relative_to_incumbent: float,
    *,
    static_analysis: StaticAnalysis | None = None,
) -> EvaluationResult:
    """Create an evaluation result that reached benchmarking."""
    score = 100.0 * relative_to_incumbent
    return EvaluationResult(
        candidate_hash=candidate_hash,
        compile_status=CompileStatus.SUCCESS,
        static_analysis=static_analysis or StaticAnalysis(registers_per_thread=48),
        correctness=CorrectnessResult(passed=True),
        benchmark=BenchmarkBundle(
            shape_results=[
                ShapeBenchResult(
                    shape_id="small",
                    latency_p50_us=score,
                    latency_p95_us=score * 1.1,
                    run_count=20,
                ),
                ShapeBenchResult(
                    shape_id="large",
                    latency_p50_us=score * 2.0,
                    latency_p95_us=score * 2.2,
                    run_count=20,
                ),
            ],
            objective_score=ObjectiveScore(
                metric_name="weighted_p50_us",
                value=score,
                relative_to_baseline=relative_to_incumbent,
                relative_to_incumbent=relative_to_incumbent,
            ),
            regressed_vs_incumbent=outcome == CandidateOutcome.REGRESSION,
        ),
        outcome=outcome,
    )


class StubLLMClient:
    """LLM client that returns configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the next configured response."""
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return self._responses[idx]


async def test_deterministic_extraction_genes_hints_and_avoid_patterns() -> None:
    """Deterministic path extracts rich fields and legacy bridges."""
    analyzer = CrossCandidateAnalyzer()
    candidate_a = _candidate(
        "hash_A",
        "__global__ void k(){ __shared__ float tile[32]; threadIdx.x; }",
    )
    candidate_b = _candidate(
        "hash_B",
        "__global__ void k(){ #pragma unroll\n for(int i=0;i<4;i++){} }",
        direction="compute_loop",
    )
    candidate_bad = _candidate(
        "hash_bad",
        "__global__ void k(){ __syncthreads(); __syncthreads(); }",
    )

    analysis = await analyzer.analyze(
        [
            (candidate_a, _result("hash_A", CandidateOutcome.IMPROVED, 0.8)),
            (candidate_b, _result("hash_B", CandidateOutcome.IMPROVED, 0.9)),
            (candidate_bad, _result("hash_bad", CandidateOutcome.REGRESSION, 1.2)),
        ],
        _make_spec(),
    )

    assert {delta.candidate_hash for delta in analysis.semantic_deltas} == {
        "hash_A",
        "hash_B",
        "hash_bad",
    }
    assert {gene.source_candidate_hash for gene in analysis.candidate_genes} == {
        "hash_A",
        "hash_B",
    }
    assert all(
        gene.source_candidate_hash != "hash_bad" for gene in analysis.candidate_genes
    )
    assert analysis.recombination_hints
    assert analysis.recombination_hints[0].parent_candidates == ["hash_A", "hash_B"]
    assert analysis.avoid_patterns
    assert analysis.avoid_patterns[0].source_candidate_hash == "hash_bad"
    assert "No evidence-backed reusable genes identified." not in analysis.winning_genes
    assert analysis.recombination_suggestions


async def test_failed_candidates_are_rejected_defensively() -> None:
    """Compile/correctness/error outcomes do not influence rich analysis."""
    analyzer = CrossCandidateAnalyzer()
    bad_candidate = _candidate("bad", "__global__ void k(){}")
    failed = EvaluationResult(
        candidate_hash="bad",
        compile_status=CompileStatus.COMPILE_ERROR,
        outcome=CandidateOutcome.COMPILE_FAIL,
    )

    analysis = await analyzer.analyze([(bad_candidate, failed)], _make_spec())

    assert analysis.semantic_deltas == []
    assert analysis.candidate_genes == []
    assert analysis.avoid_patterns == []


async def test_llm_retries_and_accepts_valid_json() -> None:
    """Malformed LLM JSON retries once, then valid rich entries are accepted."""
    llm_payload = {
        "semantic_deltas": [
            {
                "candidate_hash": "hash_A",
                "parent_hashes": ["base_hash"],
                "outcome": "IMPROVED",
                "summary": "shared memory tiling improved measured objective",
                "changed_features": ["shared_memory"],
                "evidence_refs": ["benchmark.objective_score.relative_to_incumbent"],
                "confidence": "high",
            }
        ],
        "candidate_genes": [
            {
                "gene_id": "gene_llm_1",
                "source_candidate_hash": "hash_A",
                "gene_type": "tiling",
                "description": "shared-memory tile reuse",
                "evidence": {"relative_to_incumbent": 0.8},
                "affected_shape_ids": ["large"],
                "risk_flags": ["shape_specific"],
                "confidence": "high",
            }
        ],
        "recombination_hints": [],
        "avoid_patterns": [],
        "insights": ["LLM accepted evidence-backed gene."],
    }
    llm = StubLLMClient(["not json", json.dumps(llm_payload)])
    analyzer = CrossCandidateAnalyzer(llm_client=llm)
    candidate_a = _candidate("hash_A", "__global__ void k(){ __shared__ float s[1]; }")
    candidate_b = _candidate("hash_B", "__global__ void k(){ threadIdx.x; }")

    analysis = await analyzer.analyze(
        [
            (candidate_a, _result("hash_A", CandidateOutcome.IMPROVED, 0.8)),
            (candidate_b, _result("hash_B", CandidateOutcome.BASELINE_MATCH, 1.0)),
        ],
        _make_spec(),
    )

    assert llm.call_count == 2
    assert analysis.candidate_genes[0].gene_id == "gene_llm_1"
    assert "LLM accepted evidence-backed gene." in analysis.insights


async def test_llm_fake_gene_evidence_key_falls_back_to_deterministic() -> None:
    """LLM gene claims need measured evidence keys, not source-only counts."""
    invalid_payload = {
        "semantic_deltas": [],
        "candidate_genes": [
            {
                "gene_id": "fake_count_gene",
                "source_candidate_hash": "hash_A",
                "gene_type": "tiling",
                "description": "shared-memory pattern count",
                "evidence": {"source_pattern_count": 1.0},
                "affected_shape_ids": ["large"],
                "risk_flags": [],
                "confidence": "high",
            }
        ],
        "recombination_hints": [],
        "avoid_patterns": [],
        "insights": ["fake source-count gene accepted"],
    }
    llm = StubLLMClient([json.dumps(invalid_payload), json.dumps(invalid_payload)])
    analyzer = CrossCandidateAnalyzer(llm_client=llm)
    candidate_a = _candidate("hash_A", "__global__ void k(){ __shared__ float s[1]; }")
    candidate_b = _candidate("hash_B", "__global__ void k(){ threadIdx.x; }")

    analysis = await analyzer.analyze(
        [
            (candidate_a, _result("hash_A", CandidateOutcome.IMPROVED, 0.8)),
            (candidate_b, _result("hash_B", CandidateOutcome.BASELINE_MATCH, 1.0)),
        ],
        _make_spec(),
    )

    assert llm.call_count == 2
    assert all(gene.gene_id != "fake_count_gene" for gene in analysis.candidate_genes)
    assert "fake source-count gene accepted" not in analysis.insights
    assert analysis.semantic_deltas


async def test_llm_fake_avoid_evidence_key_falls_back_to_deterministic() -> None:
    """LLM avoid claims need measured evidence keys, not source-only counts."""
    invalid_payload = {
        "semantic_deltas": [],
        "candidate_genes": [],
        "recombination_hints": [],
        "avoid_patterns": [
            {
                "pattern_id": "fake_count_avoid",
                "source_candidate_hash": "hash_bad",
                "pattern": "extra_sync",
                "reason": "sync appears in source",
                "evidence": {"source_pattern_count": 2.0},
                "affected_shape_ids": ["large"],
                "scope": "candidate_local",
                "confidence": "high",
            }
        ],
        "insights": ["fake source-count avoid accepted"],
    }
    llm = StubLLMClient([json.dumps(invalid_payload), json.dumps(invalid_payload)])
    analyzer = CrossCandidateAnalyzer(llm_client=llm)
    candidate_a = _candidate("hash_A", "__global__ void k(){ __shared__ float s[1]; }")
    candidate_bad = _candidate(
        "hash_bad",
        "__global__ void k(){ __syncthreads(); __syncthreads(); }",
    )

    analysis = await analyzer.analyze(
        [
            (candidate_a, _result("hash_A", CandidateOutcome.IMPROVED, 0.8)),
            (candidate_bad, _result("hash_bad", CandidateOutcome.REGRESSION, 1.2)),
        ],
        _make_spec(),
    )

    assert llm.call_count == 2
    assert all(
        pattern.pattern_id != "fake_count_avoid" for pattern in analysis.avoid_patterns
    )
    assert "fake source-count avoid accepted" not in analysis.insights
    assert analysis.avoid_patterns


@pytest.mark.parametrize(
    "evidence_key,evidence_value",
    [
        ("shape.large.weight", 1.0),
        ("shape.large.run_count", 20.0),
    ],
)
async def test_llm_gene_metadata_only_shape_evidence_falls_back(
    evidence_key: str,
    evidence_value: float,
) -> None:
    """LLM genes need performance evidence, not shape metadata alone."""
    invalid_payload = {
        "semantic_deltas": [],
        "candidate_genes": [
            {
                "gene_id": f"metadata_only_gene_{evidence_key}",
                "source_candidate_hash": "hash_A",
                "gene_type": "tiling",
                "description": "shape metadata is not performance evidence",
                "evidence": {evidence_key: evidence_value},
                "affected_shape_ids": ["large"],
                "risk_flags": [],
                "confidence": "high",
            }
        ],
        "recombination_hints": [],
        "avoid_patterns": [],
        "insights": ["metadata-only gene accepted"],
    }
    llm = StubLLMClient([json.dumps(invalid_payload), json.dumps(invalid_payload)])
    analyzer = CrossCandidateAnalyzer(llm_client=llm)
    candidate_a = _candidate("hash_A", "__global__ void k(){ __shared__ float s[1]; }")
    candidate_b = _candidate("hash_B", "__global__ void k(){ threadIdx.x; }")

    analysis = await analyzer.analyze(
        [
            (candidate_a, _result("hash_A", CandidateOutcome.IMPROVED, 0.8)),
            (candidate_b, _result("hash_B", CandidateOutcome.BASELINE_MATCH, 1.0)),
        ],
        _make_spec(),
    )

    assert llm.call_count == 2
    assert all(
        not gene.gene_id.startswith("metadata_only_gene_")
        for gene in analysis.candidate_genes
    )
    assert "metadata-only gene accepted" not in analysis.insights
    assert analysis.semantic_deltas


@pytest.mark.parametrize(
    "evidence_key,evidence_value",
    [
        ("shape.large.weight", 1.0),
        ("shape.large.run_count", 20.0),
    ],
)
async def test_llm_avoid_metadata_only_shape_evidence_falls_back(
    evidence_key: str,
    evidence_value: float,
) -> None:
    """LLM avoid patterns need performance evidence, not shape metadata alone."""
    invalid_payload = {
        "semantic_deltas": [],
        "candidate_genes": [],
        "recombination_hints": [],
        "avoid_patterns": [
            {
                "pattern_id": f"metadata_only_avoid_{evidence_key}",
                "source_candidate_hash": "hash_bad",
                "pattern": "extra_sync",
                "reason": "shape metadata is not regression evidence",
                "evidence": {evidence_key: evidence_value},
                "affected_shape_ids": ["large"],
                "scope": "candidate_local",
                "confidence": "high",
            }
        ],
        "insights": ["metadata-only avoid accepted"],
    }
    llm = StubLLMClient([json.dumps(invalid_payload), json.dumps(invalid_payload)])
    analyzer = CrossCandidateAnalyzer(llm_client=llm)
    candidate_a = _candidate("hash_A", "__global__ void k(){ __shared__ float s[1]; }")
    candidate_bad = _candidate(
        "hash_bad",
        "__global__ void k(){ __syncthreads(); __syncthreads(); }",
    )

    analysis = await analyzer.analyze(
        [
            (candidate_a, _result("hash_A", CandidateOutcome.IMPROVED, 0.8)),
            (candidate_bad, _result("hash_bad", CandidateOutcome.REGRESSION, 1.2)),
        ],
        _make_spec(),
    )

    assert llm.call_count == 2
    assert all(
        not pattern.pattern_id.startswith("metadata_only_avoid_")
        for pattern in analysis.avoid_patterns
    )
    assert "metadata-only avoid accepted" not in analysis.insights
    assert analysis.avoid_patterns


async def test_invalid_llm_source_only_claim_falls_back_to_deterministic() -> None:
    """Source-only LLM gene claims are rejected and deterministic output remains."""
    invalid_payload = {
        "semantic_deltas": [],
        "candidate_genes": [
            {
                "gene_id": "source_only",
                "source_candidate_hash": "hash_A",
                "gene_type": "tiling",
                "description": "has __shared__",
                "evidence": {},
                "affected_shape_ids": [],
                "risk_flags": [],
                "confidence": "high",
            }
        ],
        "recombination_hints": [],
        "avoid_patterns": [],
        "insights": [],
    }
    llm = StubLLMClient([json.dumps(invalid_payload), json.dumps(invalid_payload)])
    analyzer = CrossCandidateAnalyzer(llm_client=llm)
    candidate_a = _candidate("hash_A", "__global__ void k(){ __shared__ float s[1]; }")
    candidate_b = _candidate("hash_B", "__global__ void k(){ threadIdx.x; }")

    analysis = await analyzer.analyze(
        [
            (candidate_a, _result("hash_A", CandidateOutcome.IMPROVED, 0.8)),
            (candidate_b, _result("hash_B", CandidateOutcome.BASELINE_MATCH, 1.0)),
        ],
        _make_spec(),
    )

    assert llm.call_count == 2
    assert all(gene.gene_id != "source_only" for gene in analysis.candidate_genes)
    assert analysis.semantic_deltas
    assert analysis.winning_genes
