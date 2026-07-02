"""Unit tests for extracting grounding blocks from MinerU middle.json.

magic-pdf is not installed in the test env, so we test the pure parser
(`mineru_blocks_from_middle`) against synthetic middle.json structures. It must
yield reading-order blocks with 1-based page and bbox normalized to [0,1]
fractions of page size (top-left origin).
"""

from __future__ import annotations

import json

from nanoresearch.rag.libs.loader.mineru_loader import (
    MinerULoader,
    mineru_blocks_from_middle,
)


def test_extracts_text_page_and_normalized_bbox():
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [600.0, 800.0],
                "para_blocks": [
                    {
                        "type": "text",
                        "bbox": [60, 80, 540, 160],
                        "lines": [
                            {"spans": [{"content": "Hello"}, {"content": "world"}]}
                        ],
                    }
                ],
            }
        ]
    }

    assert mineru_blocks_from_middle(middle) == [
        {"text": "Hello world", "page": 1, "bbox": [0.1, 0.1, 0.9, 0.2]},
    ]


def test_multiple_pages_accumulate_in_order_with_1based_page():
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 100],
                "para_blocks": [
                    {"bbox": [0, 0, 50, 50], "lines": [{"spans": [{"content": "A"}]}]}
                ],
            },
            {
                "page_idx": 1,
                "page_size": [100, 100],
                "para_blocks": [
                    {"bbox": [50, 50, 100, 100], "lines": [{"spans": [{"content": "B"}]}]}
                ],
            },
        ]
    }

    blocks = mineru_blocks_from_middle(middle)

    assert [(b["text"], b["page"]) for b in blocks] == [("A", 1), ("B", 2)]


def test_nested_blocks_text_is_extracted_for_table_image_blocks():
    # Table/image blocks nest their text one level deeper under "blocks".
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 100],
                "para_blocks": [
                    {
                        "type": "table",
                        "bbox": [10, 10, 90, 40],
                        "blocks": [
                            {"lines": [{"spans": [{"content": "Caption text"}]}]}
                        ],
                    }
                ],
            }
        ]
    }

    blocks = mineru_blocks_from_middle(middle)

    assert blocks == [
        {"text": "Caption text", "page": 1, "bbox": [0.1, 0.1, 0.9, 0.4]},
    ]


def test_block_without_bbox_is_skipped():
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 100],
                "para_blocks": [
                    {"lines": [{"spans": [{"content": "no bbox"}]}]},
                    {"bbox": [0, 0, 100, 50], "lines": [{"spans": [{"content": "kept"}]}]},
                ],
            }
        ]
    }

    blocks = mineru_blocks_from_middle(middle)

    assert [b["text"] for b in blocks] == ["kept"]


def test_empty_text_block_is_skipped():
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 100],
                "para_blocks": [
                    {"type": "image", "bbox": [0, 0, 100, 50], "lines": []},
                    {"bbox": [0, 60, 100, 100], "lines": [{"spans": [{"content": "text"}]}]},
                ],
            }
        ]
    }

    blocks = mineru_blocks_from_middle(middle)

    assert [b["text"] for b in blocks] == ["text"]


def test_missing_or_zero_page_size_skips_page():
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [0, 0],
                "para_blocks": [
                    {"bbox": [0, 0, 10, 10], "lines": [{"spans": [{"content": "x"}]}]}
                ],
            },
            {
                "page_idx": 1,
                "para_blocks": [
                    {"bbox": [0, 0, 10, 10], "lines": [{"spans": [{"content": "y"}]}]}
                ],
            },
        ]
    }

    assert mineru_blocks_from_middle(middle) == []


def test_span_text_fallback_key():
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [100, 100],
                "para_blocks": [
                    {"bbox": [0, 0, 100, 100], "lines": [{"spans": [{"text": "legacy"}]}]}
                ],
            }
        ]
    }

    assert mineru_blocks_from_middle(middle)[0]["text"] == "legacy"


def test_empty_or_missing_pdf_info_is_empty():
    assert mineru_blocks_from_middle({}) == []
    assert mineru_blocks_from_middle({"pdf_info": []}) == []


def test_blocks_from_pipe_result_parses_get_middle_json():
    class FakePipe:
        def get_middle_json(self):
            return json.dumps(
                {
                    "pdf_info": [
                        {
                            "page_idx": 0,
                            "page_size": [100, 100],
                            "para_blocks": [
                                {"bbox": [0, 0, 100, 50], "lines": [{"spans": [{"content": "hi"}]}]}
                            ],
                        }
                    ]
                }
            )

    assert MinerULoader._blocks_from_pipe_result(FakePipe()) == [
        {"text": "hi", "page": 1, "bbox": [0.0, 0.0, 1.0, 0.5]},
    ]


def test_blocks_from_pipe_result_graceful_on_error():
    class BadPipe:
        def get_middle_json(self):
            raise RuntimeError("boom")

    assert MinerULoader._blocks_from_pipe_result(BadPipe()) == []
