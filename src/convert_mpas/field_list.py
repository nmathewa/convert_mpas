from __future__ import annotations

from pathlib import Path

from .models import FieldList, InputField


def _load_field_list(path: Path) -> FieldList:
    fields = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return FieldList(num_fields=len(fields), field_names=fields)


def field_list_init() -> tuple[FieldList, FieldList]:
    include_path = Path("include_fields")
    exclude_path = Path("exclude_fields")
    include_list = _load_field_list(include_path) if include_path.exists() else FieldList()
    exclude_list = _load_field_list(exclude_path) if exclude_path.exists() else FieldList()
    return include_list, exclude_list


def field_list_finalize(include_list: FieldList, exclude_list: FieldList) -> None:
    include_list.num_fields = 0
    include_list.field_names.clear()
    exclude_list.num_fields = 0
    exclude_list.field_names.clear()


def should_remap_field(field: InputField, include_list: FieldList, exclude_list: FieldList) -> bool:
    if include_list.num_fields == 0 and exclude_list.num_fields == 0:
        return True
    if include_list.num_fields != 0:
        return field.name.strip() in {name.strip() for name in include_list.field_names}
    return field.name.strip() not in {name.strip() for name in exclude_list.field_names}
