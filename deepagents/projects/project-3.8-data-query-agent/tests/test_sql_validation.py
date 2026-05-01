from data_query_agent.workflow import validate_sql


def test_validate_sql_allows_simple_select() -> None:
    assert validate_sql("SELECT * FROM tickets") is True


def test_validate_sql_blocks_mutations() -> None:
    assert validate_sql("DROP TABLE tickets") is False


def test_empty_query_result_is_still_valid_select() -> None:
    assert validate_sql("SELECT * FROM orders WHERE order_id = -1") is True
