# %%
from dataclasses import dataclass, field
from typing import Any, Iterator


# %%
@dataclass
class FunctionTemplate:
    name: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def full_params(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.params}
    
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter([
            ("name", self.name), 
            ("description", self.description), 
            ("parameters", self.full_params)
        ])


