import json


def discover(archive):
    body, _ = archive.get(
        "gdelt",
        "https://api.gdeltproject.org/api/v2/doc/doc",
        {
            "query": 'Meta ("excess compute" OR "AI computing capacity" OR "cloud business")',
            "mode": "artlist",
            "format": "json",
            "startdatetime": "20260625000000",
            "enddatetime": "20260712000000",
            "maxrecords": 250,
            "sort": "DateAsc",
        },
    )
    # seendate is discovery time, NOT a verified original publication timestamp.
    return json.loads(body).get("articles", [])
