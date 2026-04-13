"""Unit tests for QueryProcessor.

Tests the query preprocessing functionality including:
- Keyword extraction with Chinese and English tokenization
- Stopword filtering
- Filter parsing from query syntax
- Query normalization
- Edge cases (empty queries, special characters, etc.)
"""

import pytest
from nanobot.rag.core.query_engine.query_processor import (
    QueryProcessor,
    QueryProcessorConfig,
    create_query_processor,
    CHINESE_STOPWORDS,
    ENGLISH_STOPWORDS,
    DEFAULT_STOPWORDS,
)
from nanobot.rag.core.types import ProcessedQuery


class TestQueryProcessorInit:
    """Tests for QueryProcessor initialization."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        processor = QueryProcessor()
        assert processor.config.stopwords == DEFAULT_STOPWORDS
        assert processor.config.min_keyword_length == 1
        assert processor.config.max_keywords == 20
        assert processor.config.enable_filter_parsing is True

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        custom_stopwords = {"test", "custom"}
        config = QueryProcessorConfig(
            stopwords=custom_stopwords,
            min_keyword_length=2,
            max_keywords=10,
        )
        processor = QueryProcessor(config)
        assert processor.config.stopwords == custom_stopwords
        assert processor.config.min_keyword_length == 2
        assert processor.config.max_keywords == 10

    def test_factory_function(self):
        """Test create_query_processor factory function."""
        processor = create_query_processor(
            min_keyword_length=3,
            max_keywords=15,
            enable_filter_parsing=False,
        )
        assert processor.config.min_keyword_length == 3
        assert processor.config.max_keywords == 15
        assert processor.config.enable_filter_parsing is False


class TestQueryProcessorChinese:
    """Tests for Chinese query processing."""

    def test_chinese_query_basic(self):
        """Test basic Chinese query processing."""
        processor = QueryProcessor()
        result = processor.process("如何配置 Azure OpenAI？")

        assert result.original_query == "如何配置 Azure OpenAI？"
        assert "配置" in result.keywords
        assert "Azure" in result.keywords
        assert "OpenAI" in result.keywords
        # Stopwords should be filtered
        assert "如何" not in result.keywords
        assert "？" not in result.keywords

    def test_chinese_query_removes_question_words(self):
        """Test that Chinese question words are filtered."""
        processor = QueryProcessor()
        result = processor.process("怎么使用 Python？")
        # "怎么" and "使用" are stopwords, only Python should remain
        assert "Python" in result.keywords
        assert len(result.keywords) == 1

    def test_chinese_query_removes_particles(self):
        """Test that Chinese particles are filtered."""
        processor = QueryProcessor()
        result = processor.process("这个文档的内容")
        assert "这个" not in result.keywords
        assert "的" not in result.keywords


class TestQueryProcessorEnglish:
    """Tests for English query processing."""

    def test_english_query_basic(self):
        """Test basic English query processing."""
        processor = QueryProcessor()
        result = processor.process("How to configure Azure OpenAI")

        assert "configure" in result.keywords
        assert "Azure" in result.keywords
        assert "OpenAI" in result.keywords
        # Articles and question words filtered
        assert "How" not in result.keywords
        assert "to" not in result.keywords

    def test_english_query_removes_articles(self):
        """Test that English articles are filtered."""
        processor = QueryProcessor()
        result = processor.process("the configuration guide")
        assert "the" not in result.keywords
        assert "configuration" in result.keywords
        assert "guide" in result.keywords

    def test_english_query_case_preserved(self):
        """Test that keyword case is preserved."""
        processor = QueryProcessor()
        result = processor.process("Python Programming Language")
        assert "Python" in result.keywords
        assert "Programming" in result.keywords
        assert "Language" in result.keywords


class TestQueryProcessorMixed:
    """Tests for mixed Chinese/English queries."""

    def test_mixed_query(self):
        """Test processing mixed Chinese/English queries."""
        processor = QueryProcessor()
        result = processor.process("Python 教程 PDF 下载")

        assert "Python" in result.keywords
        assert "教程" in result.keywords
        assert "PDF" in result.keywords
        assert "下载" in result.keywords
        # Stopwords filtered
        assert "的" not in result.keywords

    def test_query_with_numbers(self):
        """Test query containing numbers."""
        processor = QueryProcessor()
        result = processor.process("Python 教程 3.11")
        assert "Python" in result.keywords
        assert "教程" in result.keywords
        # "3.11" might be filtered as punctuation or kept
        assert len(result.keywords) >= 2


class TestQueryProcessorFilters:
    """Tests for filter parsing."""

    def test_parse_collection_filter(self):
        """Test parsing collection filter from query."""
        processor = QueryProcessor()
        result = processor.process("Azure 文档 collection:api-docs")

        assert "Azure" in result.keywords
        assert result.filters["collection"] == "api-docs"

    def test_parse_doc_type_filter(self):
        """Test parsing document type filter."""
        processor = QueryProcessor()
        result = processor.process("报告 type:pdf")
        assert "报告" in result.keywords
        assert result.filters["doc_type"] == "pdf"

    def test_parse_multiple_filters(self):
        """Test parsing multiple filters."""
        processor = QueryProcessor()
        result = processor.process("文档 collection:docs type:pdf")
        assert result.filters["collection"] == "docs"
        assert result.filters["doc_type"] == "pdf"

    def test_parse_filter_short_keys(self):
        """Test parsing filters with short keys."""
        processor = QueryProcessor()
        result = processor.process("内容 c:default s:report.pdf")
        assert result.filters["collection"] == "default"
        assert result.filters["source_path"] == "report.pdf"

    def test_filter_parsing_disabled(self):
        """Test that filter parsing can be disabled."""
        processor = QueryProcessor(config=QueryProcessorConfig(enable_filter_parsing=False))
        result = processor.process("内容 collection:docs")
        assert "collection" in result.keywords
        assert "docs" in result.keywords
        assert result.filters == {}

    def test_parse_tag_filter(self):
        """Test parsing tag filter with comma-separated values."""
        processor = QueryProcessor()
        result = processor.process("内容 tags:python,machine-learning")
        assert "python" in result.filters["tags"]
        assert "machine-learning" in result.filters["tags"]


class TestQueryProcessorNormalization:
    """Tests for query normalization."""

    def test_whitespace_normalization(self):
        """Test that whitespace is normalized."""
        processor = QueryProcessor()
        result = processor.process("Python    教程")
        assert "Python" in result.keywords
        assert "教程" in result.keywords

    def test_unicode_normalization(self):
        """Test unicode handling."""
        processor = QueryProcessor()
        # Full-width and half-width characters
        result = processor.process("Python　教程")  # Full-width space
        assert "Python" in result.keywords
        assert "教程" in result.keywords


class TestQueryProcessorEdgeCases:
    """Tests for edge cases."""

    def test_empty_query(self):
        """Test handling of empty query."""
        processor = QueryProcessor()
        result = processor.process("")

        assert result.original_query == ""
        assert result.keywords == []
        assert result.filters == {}

    def test_whitespace_only_query(self):
        """Test handling of whitespace-only query."""
        processor = QueryProcessor()
        result = processor.process("   \t\n  ")

        assert result.keywords == []
        assert result.filters == {}

    def test_stopword_only_query(self):
        """Test query containing only stopwords."""
        processor = QueryProcessor()
        result = processor.process("的 地 得 了")

        assert result.keywords == []

    def test_punctuation_only(self):
        """Test query with only punctuation."""
        processor = QueryProcessor()
        result = processor.process("！？。，")

        assert result.keywords == []

    def test_max_keywords_limit(self):
        """Test that max_keywords limit is respected."""
        processor = QueryProcessor(config=QueryProcessorConfig(max_keywords=5))
        result = processor.process(" ".join([f"word{i}" for i in range(20)]))

        assert len(result.keywords) <= 5

    def test_keyword_deduplication(self):
        """Test that duplicate keywords are removed."""
        processor = QueryProcessor()
        result = processor.process("Python Python python")

        # Case-insensitive deduplication, preserving first occurrence's case
        python_count = sum(1 for kw in result.keywords if kw.lower() == "python")
        assert python_count == 1

    def test_min_keyword_length(self):
        """Test minimum keyword length filtering."""
        processor = QueryProcessor(config=QueryProcessorConfig(min_keyword_length=3))
        result = processor.process("Python programming tutorial ab")

        assert "Python" in result.keywords
        assert "programming" in result.keywords
        assert "tutorial" in result.keywords
        # "ab" is 2 chars, should be filtered
        assert "ab" not in result.keywords


class TestQueryProcessorStopwords:
    """Tests for stopword management."""

    def test_add_stopwords(self):
        """Test adding custom stopwords."""
        processor = QueryProcessor()
        processor.add_stopwords({"custom", "stopword"})

        result = processor.process("Python custom stopword tutorial")
        assert "custom" not in result.keywords
        assert "stopword" not in result.keywords
        assert "Python" in result.keywords
        assert "tutorial" in result.keywords

    def test_remove_stopwords(self):
        """Test removing stopwords."""
        processor = QueryProcessor()
        # Remove a default stopword
        processor.remove_stopwords({"Python"})

        result = processor.process("Python tutorial")
        assert "Python" in result.keywords  # Now included
        assert "tutorial" in result.keywords

    def test_add_chinese_stopwords(self):
        """Test adding Chinese stopwords."""
        processor = QueryProcessor()
        processor.add_stopwords({"测试词"})

        result = processor.process("测试词 Python")
        assert "测试词" not in result.keywords
        assert "Python" in result.keywords


class TestQueryProcessorOutput:
    """Tests for ProcessedQuery output format."""

    def test_output_is_processed_query_type(self):
        """Test that output is ProcessedQuery instance."""
        processor = QueryProcessor()
        result = processor.process("Python 教程")

        assert isinstance(result, ProcessedQuery)
        assert hasattr(result, "original_query")
        assert hasattr(result, "keywords")
        assert hasattr(result, "filters")

    def test_output_serialization(self):
        """Test ProcessedQuery serialization."""
        processor = QueryProcessor()
        result = processor.process("Python 教程")

        # Should be serializable to dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["original_query"] == "Python 教程"
        assert "Python" in result_dict["keywords"]
        assert "教程" in result_dict["keywords"]