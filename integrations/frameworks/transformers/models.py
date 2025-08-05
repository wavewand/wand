"""
Transformers Data Models

Defines data structures for Hugging Face Transformers operations
including text generation, classification, embeddings, and more.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseConfig, BaseQuery, BaseResponse


class TransformersTask(str, Enum):
    """Transformers task types."""

    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    FEATURE_EXTRACTION = "feature-extraction"
    FILL_MASK = "fill-mask"
    TEXT_TO_TEXT_GENERATION = "text2text-generation"
    CONVERSATIONAL = "conversational"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    SPEECH_RECOGNITION = "automatic-speech-recognition"
    TEXT_TO_SPEECH = "text-to-speech"


class TransformersModelType(str, Enum):
    """Transformers model types."""

    # Text Generation
    GPT2 = "gpt2"
    GPT_NEO = "EleutherAI/gpt-neo-2.7B"
    GPT_J = "EleutherAI/gpt-j-6B"
    BLOOM = "bigscience/bloom-7b1"
    LLAMA = "meta-llama/Llama-2-7b-hf"
    MISTRAL = "mistralai/Mistral-7B-v0.1"

    # Classification
    BERT = "bert-base-uncased"
    ROBERTA = "roberta-base"
    DISTILBERT = "distilbert-base-uncased"
    ALBERT = "albert-base-v2"
    ELECTRA = "google/electra-base-discriminator"

    # Embeddings
    SENTENCE_TRANSFORMERS = "sentence-transformers/all-MiniLM-L6-v2"
    E5_BASE = "intfloat/e5-base"
    BGE_BASE = "BAAI/bge-base-en"

    # Multilingual
    MBERT = "bert-base-multilingual-cased"
    XLM_ROBERTA = "xlm-roberta-base"

    # Specialized
    T5 = "t5-base"
    BART = "facebook/bart-base"
    PEGASUS = "google/pegasus-xsum"

    # Code
    CODEGEN = "Salesforce/codegen-2B-mono"
    INCODER = "facebook/incoder-1B"

    # Vision
    VIT = "google/vit-base-patch16-224"
    DEIT = "facebook/deit-base-patch16-224"

    # Speech
    WAV2VEC2 = "facebook/wav2vec2-base-960h"
    WHISPER = "openai/whisper-base"


class TransformersDevice(str, Enum):
    """Device types for model execution."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


@dataclass
class TransformersConfig(BaseConfig):
    """Transformers specific configuration."""

    # Model Configuration
    model_name: str = TransformersModelType.BERT.value
    task: Optional[TransformersTask] = None
    device: TransformersDevice = TransformersDevice.AUTO

    # Performance Configuration
    max_length: int = 512
    batch_size: int = 1
    num_return_sequences: int = 1

    # Generation Parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = True
    repetition_penalty: float = 1.0

    # Memory Management
    low_cpu_mem_usage: bool = True
    torch_dtype: str = "auto"  # auto, float16, float32, bfloat16
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Caching
    cache_dir: Optional[str] = "./models/transformers"
    use_cache: bool = True

    # API Configuration (for Hugging Face Hub)
    use_auth_token: Optional[str] = None
    revision: str = "main"

    # Pipeline Configuration
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Safety
    trust_remote_code: bool = False

    def __post_init__(self):
        super().__post_init__()


@dataclass
class TransformersQuery(BaseQuery):
    """Base Transformers query."""

    task: TransformersTask
    model_name: Optional[str] = None

    # Input data
    inputs: Union[str, List[str], Dict[str, Any]] = ""

    # Generation parameters
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    min_length: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    num_return_sequences: Optional[int] = None
    repetition_penalty: Optional[float] = None

    # Task-specific parameters
    task_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.query_text and isinstance(self.inputs, str):
            self.query_text = self.inputs


@dataclass
class TransformersTextGenerationQuery(TransformersQuery):
    """Text generation query."""

    prompt: str

    # Generation specific
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.task = TransformersTask.TEXT_GENERATION
        self.inputs = self.prompt
        if not self.query_text:
            self.query_text = self.prompt


@dataclass
class TransformersClassificationQuery(TransformersQuery):
    """Text classification query."""

    text: str
    labels: Optional[List[str]] = None  # For zero-shot classification

    def __post_init__(self):
        super().__post_init__()
        self.task = TransformersTask.TEXT_CLASSIFICATION
        self.inputs = self.text
        if not self.query_text:
            self.query_text = self.text
        if self.labels:
            self.task_kwargs["candidate_labels"] = self.labels


@dataclass
class TransformersQAQuery(TransformersQuery):
    """Question answering query."""

    question: str
    context: str

    def __post_init__(self):
        super().__post_init__()
        self.task = TransformersTask.QUESTION_ANSWERING
        self.inputs = {"question": self.question, "context": self.context}
        if not self.query_text:
            self.query_text = f"Q: {self.question} Context: {self.context[:100]}..."


@dataclass
class TransformersSummarizationQuery(TransformersQuery):
    """Summarization query."""

    text: str

    def __post_init__(self):
        super().__post_init__()
        self.task = TransformersTask.SUMMARIZATION
        self.inputs = self.text
        if not self.query_text:
            self.query_text = self.text[:100] + "..." if len(self.text) > 100 else self.text


@dataclass
class TransformersTranslationQuery(TransformersQuery):
    """Translation query."""

    text: str
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.task = TransformersTask.TRANSLATION
        self.inputs = self.text
        if not self.query_text:
            self.query_text = self.text
        if self.source_lang:
            self.task_kwargs["src_lang"] = self.source_lang
        if self.target_lang:
            self.task_kwargs["tgt_lang"] = self.target_lang


@dataclass
class TransformersEmbeddingQuery(TransformersQuery):
    """Feature extraction/embedding query."""

    text: Union[str, List[str]]

    def __post_init__(self):
        super().__post_init__()
        self.task = TransformersTask.FEATURE_EXTRACTION
        self.inputs = self.text
        if not self.query_text:
            if isinstance(self.text, str):
                self.query_text = self.text
            else:
                self.query_text = "; ".join(self.text[:3])


@dataclass
class TransformersResult:
    """Individual result from Transformers."""

    text: Optional[str] = None
    label: Optional[str] = None
    score: Optional[float] = None
    start: Optional[int] = None
    end: Optional[int] = None
    embeddings: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformersResponse(BaseResponse):
    """Transformers response."""

    task: Optional[TransformersTask] = None
    model_name: Optional[str] = None

    # Results
    results: List[TransformersResult] = field(default_factory=list)

    # Generation metadata
    generation_time_ms: float = 0
    tokens_generated: int = 0

    # Raw output (for debugging)
    raw_output: Optional[Any] = None

    def __post_init__(self):
        super().__post_init__()
        # Set content from results
        if not self.content and self.results:
            if self.results[0].text:
                self.content = self.results[0].text
            elif self.results[0].label:
                self.content = f"{self.results[0].label} ({self.results[0].score:.3f})"

    def get_text(self) -> str:
        """Get the main text result."""
        if self.results and self.results[0].text:
            return self.results[0].text
        return self.content or ""

    def get_labels(self) -> List[Dict[str, Any]]:
        """Get classification labels and scores."""
        return [{"label": r.label, "score": r.score} for r in self.results if r.label is not None]

    def get_embeddings(self) -> List[List[float]]:
        """Get embeddings."""
        return [r.embeddings for r in self.results if r.embeddings is not None]

    def get_answer(self) -> Optional[Dict[str, Any]]:
        """Get QA answer."""
        if self.results and self.results[0].text:
            return {
                "answer": self.results[0].text,
                "score": self.results[0].score,
                "start": self.results[0].start,
                "end": self.results[0].end,
            }
        return None

    @classmethod
    def from_transformers_output(
        cls, output: Any, query: TransformersQuery, generation_time_ms: float = 0
    ) -> 'TransformersResponse':
        """Create response from Transformers output."""
        response = cls(
            success=True,
            framework="transformers",
            task=query.task,
            model_name=query.model_name,
            generation_time_ms=generation_time_ms,
            raw_output=output,
        )

        # Parse different output types
        if isinstance(output, list):
            for item in output:
                result = TransformersResult()

                if isinstance(item, dict):
                    # Classification, QA, etc.
                    result.text = item.get("generated_text") or item.get("answer")
                    result.label = item.get("label")
                    result.score = item.get("score")
                    result.start = item.get("start")
                    result.end = item.get("end")
                    result.metadata = {
                        k: v
                        for k, v in item.items()
                        if k not in ["generated_text", "answer", "label", "score", "start", "end"]
                    }

                elif isinstance(item, str):
                    # Simple text output
                    result.text = item

                elif hasattr(item, '__len__') and not isinstance(item, str):
                    # Embeddings
                    result.embeddings = list(item)

                response.results.append(result)

        elif isinstance(output, dict):
            # Single result
            result = TransformersResult()
            result.text = output.get("generated_text") or output.get("answer")
            result.label = output.get("label")
            result.score = output.get("score")
            result.start = output.get("start")
            result.end = output.get("end")
            result.metadata = {
                k: v
                for k, v in output.items()
                if k not in ["generated_text", "answer", "label", "score", "start", "end"]
            }
            response.results.append(result)

        elif isinstance(output, str):
            # Simple string output
            result = TransformersResult(text=output)
            response.results.append(result)

        # Set tokens generated for text generation
        if query.task == TransformersTask.TEXT_GENERATION and response.results:
            if response.results[0].text:
                # Rough token estimation
                response.tokens_generated = len(response.results[0].text.split())

        return response
