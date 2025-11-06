# Speaker Attribution Improvements: Research & Roadmap

## Executive Summary

Current speaker diarization misattributes words between speakers due to temporal boundary misalignment and acoustic confusion. This document outlines research-backed strategies to improve attribution accuracy through:

1. **Immediate wins** (days): Better boundary alignment, tuning parameters
2. **Medium-term** (weeks): Feedback loops, post-processing refinement
3. **Advanced** (months): LLM-based correction, active learning systems

---

## Current Architecture Analysis

### What We Have
- **Diarization**: pyannote.audio 4.0 (state-of-the-art clustering)
- **Recognition**: SpeechBrain ECAPA-TDNN embeddings (1.71% EER)
- **Transcription**: MLX Whisper (fast, accurate)
- **Learning**: Dictionary corrections & vocabulary (text-only)

### Root Causes of Misattribution

1. **Temporal Boundary Mismatch**
   - Whisper and pyannote use different timing systems
   - Whisper timestamps can be off by several seconds
   - Diarization boundaries don't align with word boundaries

2. **Overlapping Speech**
   - Multiple speakers talking simultaneously
   - Current system assigns to single speaker based on max overlap

3. **Speaker Confusion**
   - Similar voices get mixed up
   - Short segments harder to classify accurately

4. **Context Loss**
   - No semantic understanding of speaker patterns
   - Can't use conversational context to refine attribution

---

## 🔬 2025 Cutting-Edge Research

### Latest State-of-the-Art Approaches (Published 2025)

#### 1. **SpeakerLM** (Aug 2025) - Multimodal LLM End-to-End
**Status**: Research paper | **Maturity**: Experimental | **Impact**: Revolutionary

The first unified multimodal large language model for Speaker Diarization and Recognition (SDR):

```
Key Innovation:
- Joint "who spoke when and what" in single end-to-end model
- Eliminates cascaded pipeline (no error propagation)
- Handles overlapping speech naturally
- Flexible speaker registration (anonymous/personalized modes)
- Outperforms all baselines on public benchmarks
```

**Practical applicability**: 🟡 Medium-term (6-12 months)
- Requires significant model size and compute
- Training data requirements are high
- Would replace entire current pipeline
- Best for: Large-scale deployments with compute resources

**Reference**: [arXiv:2508.06372](https://arxiv.org/abs/2508.06372)

---

#### 2. **Mamba-Based Diarization** (Oct 2024, ICASSP 2025)
**Status**: State-of-the-art | **Maturity**: Production-ready | **Impact**: Very High

Uses state space models (Mamba) instead of Transformers:

```
Advantages over Transformers:
- ✅ Better for long-form audio (lower memory)
- ✅ RNN-like with attention-like capabilities
- ✅ Longer local windows = better embeddings
- ✅ SOTA on RAMC, AISHELL, MSDWILD datasets
- ✅ Superior to both RNN and attention models
```

**Practical applicability**: 🟢 Ready now (2-3 weeks)
- Drop-in replacement for current models
- Better memory efficiency for long recordings
- Open implementations available
- Best for: Your use case (long meetings, conferences)

**Implementation path**:
```python
# Replace pyannote with Mamba-based diarization
# Available implementations:
# 1. mamba-diar (research code)
# 2. Can integrate into pyannote pipeline
# 3. Standalone inference scripts available

pip install mamba-ssm
# Follow ICASSP 2025 paper implementation
```

**Reference**: [arXiv:2410.06459](https://arxiv.org/abs/2410.06459)

---

#### 3. **Conformer with Speaker Attractors** (Jun 2025)
**Status**: Latest research | **Maturity**: Research | **Impact**: High

Convolution-augmented Transformers for local dependency modeling:

```
Key features:
- Conformer (CNN + Transformer) instead of pure Transformer
- Models both local and global dependencies
- Speaker attribute attractors for consistency
- Better on speech tasks needing local temporal info
- Improved CALLHOME performance
```

**Practical applicability**: 🟡 Medium-term (3-6 months)
- Research code available but needs engineering
- Could be integrated into pyannote
- Best for: High-accuracy requirements

**Reference**: [arXiv:2506.05593](https://arxiv.org/abs/2506.05593)

---

#### 4. **Multi-Channel Sequence-to-Sequence** (May 2025, MISP Challenge Winner)
**Status**: Competition winner | **Maturity**: Research | **Impact**: High

Achieves 8.09% DER (1st place MISP 2025 Challenge):

```
Approach:
- Multi-channel audio processing
- Sequence-to-sequence neural diarization
- Automatic speaker detection
- State-of-the-art results
```

**Practical applicability**: 🔴 Not yet (single-channel focus)
- Requires multi-microphone setup
- Complex architecture
- Best for: Multi-mic conference rooms (future)

**Reference**: [arXiv:2505.16387](https://arxiv.org/abs/2505.16387)

---

#### 5. **Whisper Large V3 Turbo** (Feb 2025 integrations)
**Status**: Production | **Maturity**: Ready | **Impact**: Medium-High

Latest Whisper model with 5.4x speedup:

```
Benefits:
- 4 decoder layers (vs 32) = much faster
- Similar accuracy to Large V2
- Now integrated with latest diarization:
  * Replicate's whisper-diarization (uses V3 Turbo)
  * faster-whisper 1.1.1 + pyannote 3.3.1
  * whisply supports V3 Turbo + diarization
```

**Practical applicability**: 🟢 Ready now (days)
- Can replace MLX Whisper
- Faster with similar quality
- Good diarization integrations available
- Best for: Speed improvements

**Migration path**:
```bash
# Option 1: faster-whisper + pyannote (like current setup)
pip install faster-whisper==1.1.1

# Option 2: whisply (GUI tool)
pip install whisply

# Option 3: Replicate API
# curl replicate.com/thomasmol/whisper-diarization
```

---

## Improvement Strategies (Prioritized)

### 🚀 Phase 1: Quick Wins (Days of work)

#### 1.1 WhisperX Integration
**Impact**: High | **Effort**: Medium | **Research**: Proven

Replace standard Whisper with WhisperX for better temporal alignment:

```bash
# Benefits
- Word-level timestamps instead of utterance-level
- VAD pre-segmentation for cleaner boundaries
- Forced alignment with phoneme model
- 3-5 second improvement in timestamp accuracy
```

**Implementation**:
- Install whisperx: `pip install whisperx`
- Replace MLX Whisper call with WhisperX
- WhisperX includes built-in diarization (alternative to pyannote)

**Reference**: [WhisperX (2023)](https://github.com/m-bain/whisperX)

#### 1.2 Pyannote Precision-2 Upgrade
**Impact**: High | **Effort**: Low | **Research**: Proven

Upgrade to latest pyannote Precision-2 model:

```python
# Current: pyannote/speaker-diarization-3.1
# Upgrade to: pyannote/speaker-diarization-4.0 (Precision-2)

# Benefits:
# - 15% relative improvement over open-source baseline
# - 70% accuracy on difficult benchmarks (vs 50% for Precision-1)
# - Better speaker count prediction
# - STT reconciliation flag for single-speaker-at-time mode
```

**Implementation**:
```python
# In _load_diarization_pipeline():
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-4.0",  # Updated model
    use_auth_token=self.hf_token
)

# Enable STT reconciliation for cleaner boundaries
pipeline.instantiate({
    "sttr_reconciliation": True  # Only one speaker active at a time
})
```

**Reference**: [Pyannote Precision-2 (2024)](https://www.pyannote.ai/blog/precision-2)

#### 1.3 Boundary Smoothing & Refinement
**Impact**: Medium | **Effort**: Low | **Research**: Standard practice

Add post-processing to align diarization boundaries with word boundaries:

```python
def _align_boundaries_with_words(
    segments: List[TranscriptionSegment],
    diarization: Annotation
) -> List[TranscriptionSegment]:
    """
    Align diarization speaker boundaries to word boundaries.

    Strategy:
    1. When diarization boundary falls mid-word, expand to word boundary
    2. Use word confidence scores to resolve ambiguous regions
    3. Apply minimum segment duration (e.g., 0.5s)
    """
    for segment in segments:
        # Find diarization speakers overlapping this word
        overlaps = find_overlapping_speakers(segment, diarization)

        # If boundary is mid-word, check neighboring words
        if len(overlaps) > 1:
            # Look at context: who spoke before/after?
            segment.speaker = resolve_with_context(
                segment, overlaps, segments
            )
```

**Benefits**:
- Reduces misattribution at speaker change points
- Uses word timing to guide boundary placement
- Low computational cost

#### 1.4 Parameter Tuning
**Impact**: Medium | **Effort**: Low | **Research**: Best practice

Tune diarization parameters for your use case:

```python
# Current: Uses defaults
# Better: Tune for specific audio characteristics

pipeline = Pipeline.from_pretrained(...)
pipeline.instantiate({
    # Segmentation
    "segmentation": {
        "min_duration_off": 0.0,  # Minimum silence between speakers
        "threshold": 0.5,  # Voice activity detection threshold
    },

    # Clustering
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 15,  # Minimum segments per speaker
        "threshold": 0.7,  # Similarity threshold for clustering
    },

    # STT Integration
    "sttr_reconciliation": True,  # Force single speaker at a time
})
```

**Tuning approach**:
1. Start with defaults
2. Process test recordings
3. Analyze errors (too many/few speakers? boundary issues?)
4. Adjust parameters iteratively

---

### 🎯 Phase 2: Feedback Loops (Weeks of work)

#### 2.1 Speaker Attribution Correction Learning
**Impact**: High | **Effort**: Medium | **Research**: Proven (36.5% DER reduction)

Extend the existing `CorrectionLearner` to learn from speaker attribution corrections:

```python
# New module: speaker_learning.py

class SpeakerAttributionLearner:
    """Learn from manual speaker corrections."""

    def learn_from_corrections(
        self,
        original_transcript: str,
        corrected_transcript: str,
        audio_path: Path
    ) -> SpeakerLearningResult:
        """
        Analyze speaker corrections and improve future diarization.

        Learning strategies:
        1. Boundary adjustments: Where do boundaries get moved?
        2. Context patterns: What phrases indicate speaker changes?
        3. Acoustic patterns: Extract embeddings from corrected segments
        4. Confusion matrix: Which speakers get confused?
        """

        # 1. Detect speaker changes
        changes = self._diff_speaker_labels(original, corrected)

        # 2. Extract acoustic segments for confused speakers
        confusion_pairs = self._find_confusion_pairs(changes)

        # 3. Re-enroll speakers with corrected segments
        for speaker, segments in confusion_pairs.items():
            self._update_speaker_profile(speaker, segments, audio_path)

        # 4. Learn boundary patterns
        boundary_patterns = self._learn_boundary_patterns(changes)

        return SpeakerLearningResult(
            changes=changes,
            confusion_matrix=confusion_pairs,
            boundary_adjustments=boundary_patterns
        )
```

**Workflow**:
```bash
# 1. Transcribe video
python examples/transcribe_only.py meeting.mp4

# 2. Manually correct speaker labels
#    SPEAKER_00: Hello → Alexander: Hello
#    SPEAKER_01: Hi → Sarah: Hi

# 3. Learn from corrections
python examples/learn_speaker_corrections.py \
    output/meeting_transcript.txt \
    output/meeting_corrected.txt \
    meeting.mp4

# The system will:
# - Update speaker embeddings with corrected segments
# - Learn which phrases indicate speaker changes
# - Build confusion matrices to avoid future errors
```

**Benefits**:
- System gets smarter with each correction
- Adapts to your specific speakers and scenarios
- Proven 36.5% relative DER improvement

**Reference**: [Active Correction (2022)](https://www.mdpi.com/2076-3417/12/4/1782)

#### 2.2 Confidence Scoring & Flagging
**Impact**: Medium | **Effort**: Low | **Research**: Standard practice

Add confidence scores to identify uncertain attributions:

```python
class SegmentConfidence:
    """Calculate confidence scores for speaker attribution."""

    def score_segment(
        self,
        segment: TranscriptionSegment,
        diarization: Annotation,
        embeddings: Dict[str, np.ndarray]
    ) -> float:
        """
        Score confidence of speaker attribution.

        Factors:
        - Overlap ratio (higher = more confident)
        - Segment duration (very short = less confident)
        - Embedding similarity to known speaker
        - Context consistency (same speaker as neighbors?)
        """

        scores = {
            'overlap': self._score_overlap(segment, diarization),
            'duration': self._score_duration(segment),
            'embedding': self._score_embedding_match(segment, embeddings),
            'context': self._score_context_consistency(segment, neighbors),
        }

        return weighted_average(scores)

# Usage in transcription
for segment in segments:
    segment.speaker = assign_speaker(segment)
    segment.confidence = scorer.score_segment(segment, ...)

    # Flag low confidence for review
    if segment.confidence < 0.6:
        segment.needs_review = True
```

**Output format**:
```json
{
  "speaker": "SPEAKER_00",
  "text": "Hello everyone",
  "confidence": 0.45,
  "needs_review": true,
  "confidence_breakdown": {
    "overlap": 0.8,
    "duration": 0.2,  // Very short segment
    "embedding": 0.5,
    "context": 0.3    // Different from neighbors
  }
}
```

**Benefits**:
- Helps users focus correction efforts
- Identifies problem areas automatically
- Can trigger manual review workflows

#### 2.3 Interactive Correction CLI
**Impact**: High | **Effort**: Medium | **Research**: Proven (44% confusion error reduction)

Build an interactive tool for quick speaker corrections:

```bash
python examples/correct_speakers.py output/meeting_transcript.txt

# Interactive session:
# ==========================================
# Segment 47 [00:02:34 - 00:02:36] (confidence: 0.45)
# SPEAKER_00: "I think that's a great idea"
#
# Options:
#   [1] Keep SPEAKER_00
#   [2] Change to SPEAKER_01
#   [3] Change to SPEAKER_02
#   [a] Auto-fix remaining with this pattern
#   [s] Skip
#   [q] Quit
#
# Your choice: 2
# ✓ Changed to SPEAKER_01
# ✓ Auto-enrolling corrected segment...
# ==========================================
```

**Features**:
- Shows low-confidence segments first
- Plays audio snippet for context
- Learns from each correction immediately
- Applies patterns automatically
- Minimal user effort

**Reference**: [Interactive Real-Time Correction (2024)](https://arxiv.org/abs/2509.18377)

---

### 🧠 Phase 3: Advanced Intelligence (Months of work)

#### 3.1 DiarizationLM Post-Processing
**Impact**: Very High | **Effort**: High | **Research**: Cutting-edge (55% WDER reduction)

Use LLMs to refine speaker attribution based on semantic understanding:

```python
# New module: llm_diarization_corrector.py

class LLMDiarizationCorrector:
    """
    Use LLM to refine speaker attribution using semantic context.

    Based on DiarizationLM (2024):
    - Analyzes conversation semantics
    - Identifies unnatural speaker switches
    - Uses conversational patterns
    - Word-level refinement
    """

    def refine_attribution(
        self,
        segments: List[TranscriptionSegment],
        context: str = ""
    ) -> List[TranscriptionSegment]:
        """
        Use LLM to refine speaker boundaries.

        Prompt engineering:
        "Given this conversation transcript, identify any unnatural
        speaker switches where a single thought is split across speakers.

        Markers of correct attribution:
        - Questions followed by answers are different speakers
        - Continuation of same topic is usually same speaker
        - Mid-sentence switches are usually errors
        - 'Yes, and' patterns indicate response to different speaker
        "
        """

        # Chunk transcript at reasonable boundaries
        chunks = self._chunk_at_natural_boundaries(segments)

        for chunk in chunks:
            # Build context-aware prompt
            prompt = self._build_correction_prompt(
                chunk,
                context=context,
                previous_chunks=previous
            )

            # Get LLM refinement
            corrections = self._query_llm(prompt)

            # Apply corrections
            chunk = self._apply_llm_corrections(chunk, corrections)

        return segments

    def _build_correction_prompt(self, chunk, context, previous):
        return f"""
        You are analyzing a conversation transcript for speaker attribution accuracy.

        Context: {context}

        Previous conversation:
        {previous}

        Current section:
        {self._format_for_llm(chunk)}

        Task: Identify any speaker attribution errors where:
        1. A single sentence is split across multiple speakers
        2. Question-answer pairs have the same speaker
        3. Natural conversation flow is broken

        For each error, provide:
        - Word range (start, end)
        - Correct speaker
        - Confidence (0-1)
        - Reasoning

        Output JSON format:
        [
          {{
            "word_range": [145, 152],
            "current_speaker": "SPEAKER_00",
            "correct_speaker": "SPEAKER_01",
            "confidence": 0.9,
            "reasoning": "Completes the question from SPEAKER_01"
          }}
        ]
        """
```

**Implementation approach**:
1. Use local Ollama with instruction-tuned model (llama3.1, qwen2.5)
2. Process in chunks to maintain context
3. Apply corrections at word level
4. Validate before applying
5. Build test set for evaluation

**Benefits**:
- Semantic understanding of conversation
- Catches logical errors diarization misses
- Proven 55.5% WDER reduction on Fisher dataset
- Works with any base diarization system

**Challenges**:
- Computationally expensive (2-5 min per hour of audio)
- Requires good prompt engineering
- May need fine-tuning for domain

**Reference**: [DiarizationLM (2024)](https://arxiv.org/abs/2401.03506)

#### 3.2 Ensemble Diarization
**Impact**: Medium-High | **Effort**: Medium | **Research**: Proven

Combine multiple diarization systems for better accuracy:

```python
class EnsembleDiarizer:
    """
    Combine multiple diarization approaches.

    Systems to ensemble:
    1. pyannote.audio (clustering-based)
    2. WhisperX built-in diarization
    3. Custom voice embedding clustering
    """

    def ensemble_diarize(
        self,
        audio_path: Path,
        transcription: TranscriptionResult
    ) -> List[TranscriptionSegment]:
        """
        Run multiple diarization systems and vote.

        Strategy:
        - Run 2-3 diarization systems in parallel
        - Align their outputs
        - Vote on boundaries and speaker assignments
        - Use confidence scores as weights
        """

        # Run systems in parallel
        diarizations = asyncio.gather(
            self._run_pyannote(audio_path),
            self._run_whisperx(audio_path),
            self._run_speechbrain(audio_path)
        )

        # Align outputs
        aligned = self._align_diarizations(diarizations)

        # Vote on attribution
        for segment in transcription.segments:
            votes = self._collect_votes(segment, aligned)
            segment.speaker = self._vote(votes)
            segment.confidence = self._vote_confidence(votes)

        return transcription.segments
```

**Benefits**:
- More robust than single system
- Reduces system-specific biases
- Higher confidence in agreements

**Costs**:
- 2-3x computation time
- More complex pipeline
- Need alignment logic

#### 3.3 Fine-Tuning on Your Data
**Impact**: Very High (for specific domains) | **Effort**: Very High | **Research**: Proven

Fine-tune pyannote models on your specific speakers/scenarios:

```python
# Requires labeled dataset of your speakers
# Process:
# 1. Collect 10-50 hours of labeled audio with your speakers
# 2. Format as pyannote training data
# 3. Fine-tune segmentation and embedding models
# 4. Evaluate on held-out test set

# Benefits:
# - Dramatic improvement for domain-specific audio
# - Adapts to acoustic environment
# - Learns speaker characteristics
# - Proven approach in pyannote documentation

# Effort:
# - Requires ML engineering expertise
# - Need labeled training data
# - GPU compute for training
# - Evaluation and iteration
```

**When to do this**: Only if processing large volumes of similar audio (same speakers, same environment) and accuracy is critical.

**Reference**: [Pyannote Fine-tuning Guide](https://github.com/pyannote/pyannote-audio)

---

## Recommended Implementation Roadmap

### Week 1: Foundation (Quick Wins)
- [ ] Upgrade to pyannote Precision-2 model
- [ ] Enable STT reconciliation flag
- [ ] Implement boundary smoothing
- [ ] Add confidence scoring
- [ ] Tune parameters on test recordings

**Expected improvement**: 10-20% reduction in misattribution errors

### Week 2-3: Feedback System
- [ ] Create `SpeakerAttributionLearner` class
- [ ] Build interactive correction CLI
- [ ] Implement speaker embedding updates
- [ ] Add confusion matrix tracking
- [ ] Create correction workflow documentation

**Expected improvement**: Additional 20-30% with active corrections

### Week 4-5: Integration
- [ ] Evaluate WhisperX vs MLX Whisper trade-offs
- [ ] Integrate best option
- [ ] Build test suite with ground truth
- [ ] Benchmark improvements
- [ ] Document parameter tuning guide

**Expected improvement**: Additional 5-15% from better alignment

### Month 2-3: Advanced (Optional)
- [ ] Implement LLM post-processing
- [ ] Build ensemble diarization
- [ ] Create automated testing framework
- [ ] Collect metrics on improvement
- [ ] Consider fine-tuning if ROI justifies

**Expected improvement**: Additional 30-50% in complex scenarios

---

## Testing & Evaluation Framework

### Metrics to Track
1. **Diarization Error Rate (DER)**: Standard metric
   - Miss rate: Speech not attributed to any speaker
   - False alarm rate: Non-speech attributed
   - Speaker confusion: Wrong speaker assigned

2. **Word Diarization Error Rate (WDER)**: More granular
   - Word-level accuracy
   - Better for transcription use case

3. **Boundary Accuracy**: How close are boundaries?
   - Mean absolute error in seconds
   - Within-tolerance rate (e.g., <0.5s)

### Test Set Creation
```bash
# Create ground truth for testing
python examples/create_test_set.py meeting.mp4

# Manually annotate:
# 1. Speaker boundaries
# 2. Speaker identities
# 3. Difficult sections

# Evaluate system:
python examples/evaluate_diarization.py \
    output/meeting_transcript.json \
    test_data/meeting_ground_truth.json

# Output:
# DER: 12.3% → 8.1% (34% improvement)
# WDER: 18.5% → 11.2% (40% improvement)
# Boundary error: 1.2s → 0.4s (67% improvement)
```

---

## Cost-Benefit Analysis

### Quick Wins (Phase 1)
- **Effort**: 2-3 days
- **Cost**: Minimal (just development time)
- **Benefit**: 10-20% improvement
- **ROI**: Very high ⭐⭐⭐⭐⭐

### Feedback Loops (Phase 2)
- **Effort**: 2-3 weeks
- **Cost**: Development + user correction time
- **Benefit**: 20-30% additional improvement
- **ROI**: High ⭐⭐⭐⭐

### LLM Processing (Phase 3a)
- **Effort**: 2-4 weeks
- **Cost**: Compute (2-5 min/hour audio) + development
- **Benefit**: 30-50% improvement in complex cases
- **ROI**: Medium-High ⭐⭐⭐

### Ensemble (Phase 3b)
- **Effort**: 2-3 weeks
- **Cost**: 2-3x compute time + development
- **Benefit**: 10-15% additional improvement
- **ROI**: Medium ⭐⭐⭐

### Fine-Tuning (Phase 3c)
- **Effort**: 1-2 months
- **Cost**: Labeled data + GPU compute + ML expertise
- **Benefit**: 40-60% improvement (domain-specific)
- **ROI**: High for specialized use cases ⭐⭐⭐⭐

---

## References & Further Reading

### Papers
1. **DiarizationLM** (2024): LLM post-processing for 55% WDER reduction
   - [arxiv.org/abs/2401.03506](https://arxiv.org/abs/2401.03506)

2. **WhisperX** (2023): Time-accurate transcription with proper alignment
   - [github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)

3. **Interactive Correction** (2024): Real-time feedback loop
   - [arxiv.org/abs/2509.18377](https://arxiv.org/abs/2509.18377)

4. **Active Learning** (2022): Human-in-loop improvement
   - [mdpi.com/2076-3417/12/4/1782](https://www.mdpi.com/2076-3417/12/4/1782)

5. **Pyannote Precision-2** (2024): Latest state-of-the-art model
   - [pyannote.ai/blog/precision-2](https://www.pyannote.ai/blog/precision-2)

### Tools & Libraries
- **pyannote.audio**: State-of-the-art diarization
- **WhisperX**: Better Whisper with alignment
- **SpeechBrain**: Speaker embeddings and recognition
- **DiarizationLM**: LLM-based correction framework

### Community Resources
- Pyannote Discord: Active community support
- Hugging Face forums: Model discussions
- arxiv.org: Latest research papers

---

## Next Steps

1. **Immediate (this week)**:
   - [ ] Review this document with stakeholders
   - [ ] Prioritize improvements based on needs
   - [ ] Set up test recordings for benchmarking
   - [ ] Create ground truth for evaluation

2. **Short-term (next 2 weeks)**:
   - [ ] Implement Phase 1 quick wins
   - [ ] Measure improvements
   - [ ] Decide on Phase 2 scope

3. **Medium-term (next month)**:
   - [ ] Build feedback loop system
   - [ ] Create user correction workflows
   - [ ] Evaluate advanced techniques

4. **Long-term (2-3 months)**:
   - [ ] Consider LLM post-processing
   - [ ] Evaluate ensemble approach
   - [ ] Assess fine-tuning ROI

---

*Document created: 2025-11-06*
*Research based on: 2023-2025 literature*
*Status: Active research & development*
