from typing import Type, TypeVar, Union
from pydantic import BaseModel

# Ensure we import the correct class definition
from src.state import CategoriesWithEvents

T = TypeVar("T", bound=BaseModel)


def ensure_pydantic_model(data: Union[dict, T], model_class: Type[T]) -> T:
    """Safely converts dict to Pydantic model."""
    if data is None:
        # Return empty instance if None
        return model_class()

    if isinstance(data, model_class):
        return data

    if isinstance(data, dict):
        try:
            return model_class(**data)
        except Exception as e:
            print(f"Pydantic Conversion Error: {e}. Data: {data}")
            # Fallback to empty to prevent crash
            return model_class()

    if hasattr(data, "__dict__"):
        return model_class(**data.__dict__)

    raise TypeError(f"Cannot convert {type(data)} to {model_class}")


def ensure_categories_with_events(
    data: Union[dict, CategoriesWithEvents, None],
) -> CategoriesWithEvents:
    """Specific helper for CategoriesWithEvents."""
    # Ensure even if None is passed, we get an empty object with defaults
    if data is None:
        return CategoriesWithEvents(context="", conflict="", reaction="", outcome="")

    return ensure_pydantic_model(data, CategoriesWithEvents)
