from tsann.types import Query, Record


def passes_filters(record: Record, query: Query) -> bool:
    return (
        query.t_start <= record.timestamp <= query.t_end
        and query.price_min <= record.price <= query.price_max
        and (query.category is None or record.category == query.category)
    )


def validate_query(query: Query) -> None:
    if query.k < 0:
        raise ValueError("Query.k must be non-negative")
    if query.t_start > query.t_end:
        raise ValueError("Query.t_start must be <= Query.t_end")
    if query.price_min > query.price_max:
        raise ValueError("Query.price_min must be <= Query.price_max")
