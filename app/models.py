from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    
class SearchResult(BaseModel):
    location: str
    start: int
    end: int
    function: str | None = None
    code: str
    score: float